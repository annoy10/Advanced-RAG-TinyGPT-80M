from workflows.state import RAGState
from generation.citation_generator import format_citations
from guardrails.refusal_handler import refuse_if_unsupported
from guardrails.relevance_checker import is_relevant
from guardrails.faithfulness_checker import check_faithfulness
from generation.context_builder import build_context
class RAGWorkflow:
    def __init__(self,retriever,generator,reranker=None,top_k=5):
        self.retriever=retriever; self.generator=generator; self.reranker=reranker; self.top_k=top_k
    def invoke(self,question):
        state=RAGState(question=question)
        docs=self.retriever.search(question,max(self.top_k*3,10))
        if self.reranker: docs=self.reranker.rerank(question,docs,self.top_k)
        else: docs=docs[:self.top_k]
        state.retrieved=docs
        refusal=refuse_if_unsupported(docs)
        if refusal or not is_relevant(question,docs):
            state.answer=refusal or "I could not find sufficiently relevant evidence in the indexed sources."
            return state
        context=build_context(docs)
        answer=self.generator.answer(question,docs)
        import re
        faith=check_faithfulness(re.sub(r"\[S\d+\]","",answer),context)
        if not faith["faithful"]:
            state.answer="I found related passages, but the available evidence was not strong enough to produce a reliably supported answer."
            return state
        state.answer=format_citations(answer,docs)
        state.citations=[d.get("metadata",{}) for d in docs]
        return state
