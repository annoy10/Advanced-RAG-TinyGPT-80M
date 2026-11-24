"""Pre-download the configured embedding and cross-encoder models."""
from config.settings import EMBEDDING_MODEL, RERANKER_MODEL
from sentence_transformers import SentenceTransformer, CrossEncoder

def main():
    print("Downloading embedding model:", EMBEDDING_MODEL)
    SentenceTransformer(EMBEDDING_MODEL)
    print("Downloading reranker model:", RERANKER_MODEL)
    CrossEncoder(RERANKER_MODEL)
    print("Models are cached and ready.")

if __name__ == "__main__":
    main()
