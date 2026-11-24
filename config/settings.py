from pathlib import Path
import os
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
def env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1","true","yes","on"}
APP_NAME = os.getenv("APP_NAME", "Advanced RAG - TinyGPT")
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STORAGE_DIR = ROOT_DIR / "storage"
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(STORAGE_DIR / "qdrant")))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEVICE = os.getenv("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "12"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "12"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
ENABLE_RERANKER = env_bool("ENABLE_RERANKER", True)
ENABLE_TINYGPT = env_bool("ENABLE_TINYGPT", False)
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.12"))

CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "parent_child").strip().lower()

USE_LANGCHAIN_RAG = env_bool("USE_LANGCHAIN_RAG", True)
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "extractive").strip().lower()
