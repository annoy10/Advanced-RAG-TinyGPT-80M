from config.settings import (ROOT_DIR, QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, DEVICE,
    RERANKER_MODEL, ENABLE_RERANKER, TOP_K_FINAL, MIN_RELEVANCE_SCORE, USE_LANGCHAIN_RAG, GENERATION_BACKEND)
from embeddings.embedding_manager import EmbeddingManager
from vectorstore.qdrant_store import QdrantStore
from vectorstore.index_manager import IndexManager
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_rewriter import QueryRewriter
from retrieval.rag_fusion import RAGFusion
from generation.llm_manager import LLMManager
from workflows.rag_graph import RAGWorkflow
from generation.langchain_rag import LangChainRAGWorkflow
from chains.ingestion_chain import ingest_sources
from config.logging_config import setup_logging
logger=setup_logging()

def build_runtime():
    embedder=EmbeddingManager(EMBEDDING_MODEL,DEVICE)
    store=QdrantStore(QDRANT_PATH,COLLECTION_NAME,embedder.dimension)
    index=IndexManager()
    records=index.load()
    bm25=BM25Retriever(records)
    vector=VectorRetriever(store,embedder)
    hybrid=HybridRetriever(vector,bm25)
    fusion=RAGFusion(hybrid,QueryRewriter())
    reranker=None
    if ENABLE_RERANKER:
        try:
            from retrieval.reranker import Reranker
            reranker=Reranker(RERANKER_MODEL,DEVICE)
            logger.info("Cross-encoder reranker enabled")
        except Exception as e:
            logger.warning("Reranker unavailable; continuing without it: %s",e)
    if USE_LANGCHAIN_RAG:
        workflow=LangChainRAGWorkflow(fusion,reranker,TOP_K_FINAL,GENERATION_BACKEND)
        logger.info('LangChain LCEL RAG workflow enabled (backend=%s)', GENERATION_BACKEND)
    else:
        workflow=RAGWorkflow(fusion,LLMManager(),reranker,TOP_K_FINAL)
    def index_records(new_records):
        if not new_records: return
        vectors=embedder.embed_documents([r["text"] for r in new_records])
        store.upsert(new_records,vectors)
        merged={r["id"]:r for r in index.load()}
        merged.update({r["id"]:r for r in new_records})
        index.save(list(merged.values()))
        # Refresh lexical index after every ingestion.
        workflow.retriever.retriever.bm25=BM25Retriever(index.load())
    return {"workflow":workflow,"index":index_records,"ingest":ingest_sources,
            "store":store,"embedder":embedder}
if __name__=="__main__":
    import gradio as gr
    from app_gradio import create_demo
    runtime=build_runtime()
    create_demo(runtime["workflow"]).launch(server_name="127.0.0.1",server_port=7860,inbrowser=True)
