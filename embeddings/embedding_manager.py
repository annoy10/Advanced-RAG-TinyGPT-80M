from config.settings import EMBEDDING_MODEL, DEVICE
from .embedding_model import EmbeddingModel
class EmbeddingManager:
    def __init__(self, model_name=EMBEDDING_MODEL, device=DEVICE):
        self.backend=EmbeddingModel(model_name,device)
        self.dimension=self.backend.dimension
    def embed_documents(self,texts): return self.backend.encode(texts)
    def embed_query(self,text): return self.backend.encode([text])[0]
