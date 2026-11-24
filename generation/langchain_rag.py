"""LangChain LCEL orchestration for the local hybrid-RAG pipeline.

The retrieval stack remains the project's own hybrid dense/BM25 + RRF implementation.
LangChain provides the document abstraction, prompt composition, and runnable chain.
The default answer backend is extractive/grounded; TinyGPT is available as an experimental
backend because the supplied TinyGPT checkpoint was trained on sports QA, not instruction data.
"""
from __future__ import annotations
from typing import Any

from workflows.state import RAGState
from generation.context_builder import build_context
from generation.citation_generator import format_citations
from generation.answer_generator import AnswerGenerator
from guardrails.refusal_handler import refuse_if_unsupported
from guardrails.relevance_checker import is_relevant
from guardrails.faithfulness_checker import check_faithfulness


class LangChainRAGWorkflow:
    def __init__(self, retriever, reranker=None, top_k=5, backend="extractive"):
        try:
            from langchain_core.documents import Document
            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
        except ImportError as exc:
            raise RuntimeError(
                "LangChain packages are missing. Run: pip install -r requirements.txt"
            ) from exc

        self.Document = Document
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.backend = backend
        self.extractive_generator = AnswerGenerator()
        self.prompt = PromptTemplate.from_template(
            "You are a careful retrieval-grounded assistant. Use only the evidence below. "
            "If evidence is insufficient, say so. Cite factual claims with [S#].\n\n"
            "QUESTION:\n{question}\n\nEVIDENCE:\n{context}\n\nANSWER:"
        )

        # LCEL pipeline: question -> hybrid retrieval -> prompt/context preparation -> grounded response.
        self.chain = (
            RunnableParallel(
                question=RunnablePassthrough(),
                documents=RunnableLambda(self._retrieve_documents),
            )
            | RunnableLambda(self._prepare_prompt)
            | RunnableLambda(self._generate_grounded)
        )

    def _retrieve_documents(self, question: str):
        raw = self.retriever.search(question, max(self.top_k * 3, 10))
        if self.reranker:
            raw = self.reranker.rerank(question, raw, self.top_k)
        else:
            raw = raw[:self.top_k]
        docs = []
        for item in raw:
            metadata = dict(item.get("metadata") or {})
            metadata.update({
                "rag_id": item.get("id", ""),
                "score": item.get("rerank_score", item.get("fusion_score", item.get("score", 0.0))),
                "parent_text": item.get("parent_text", ""),
            })
            docs.append(self.Document(page_content=item.get("text", ""), metadata=metadata))
        return docs

    def _prepare_prompt(self, payload: dict[str, Any]):
        docs = payload["documents"]
        # Keep source text and metadata in LangChain Document objects through the chain.
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name") or doc.metadata.get("source") or "unknown"
            context_parts.append(f"[S{i}] Source: {source}\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        prompt_value = self.prompt.format(question=payload["question"], context=context)
        return {
            "question": payload["question"],
            "documents": docs,
            "context": context,
            "prompt": prompt_value,
        }

    def _as_dicts(self, docs):
        converted = []
        for doc in docs:
            converted.append({
                "id": doc.metadata.get("rag_id", ""),
                "text": doc.page_content,
                "parent_text": doc.metadata.get("parent_text", ""),
                "metadata": {k:v for k,v in doc.metadata.items()
                             if k not in {"rag_id", "score", "parent_text"}},
                "score": doc.metadata.get("score", 0.0),
            })
        return converted

    def _generate_grounded(self, payload):
        question = payload["question"]
        docs = self._as_dicts(payload["documents"])
        state = RAGState(question=question, retrieved=docs)

        refusal = refuse_if_unsupported(docs)
        if refusal or not is_relevant(question, docs):
            state.answer = refusal or "I could not find sufficiently relevant evidence in the indexed sources."
            return state

        # Extractive mode is deliberately the safe default. The prompt is prepared by LCEL,
        # while the answer is selected from retrieved evidence to avoid unsupported generation.
        if self.backend == "tinygpt":
            try:
                from generation.tinygpt_adapter import TinyGPTAdapter
                answer = TinyGPTAdapter().generate(payload["prompt"])
            except Exception as exc:
                state.errors.append(f"TinyGPT backend unavailable: {exc}")
                answer = self.extractive_generator.generate(question, docs)
        else:
            answer = self.extractive_generator.generate(question, docs)

        import re
        faith = check_faithfulness(re.sub(r"\[S\d+\]", "", answer), payload["context"])
        if not faith["faithful"]:
            state.answer = (
                "I found related passages, but the available evidence was not strong enough "
                "to produce a reliably supported answer."
            )
            return state

        state.answer = format_citations(answer, docs)
        state.citations = [d.get("metadata", {}) for d in docs]
        return state

    def invoke(self, question: str):
        return self.chain.invoke(str(question).strip())
