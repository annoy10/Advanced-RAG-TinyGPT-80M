from .qdrant_store import QdrantStore
def open_store(path,name,dimension): return QdrantStore(path,name,dimension)
