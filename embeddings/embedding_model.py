class EmbeddingModel:
    def __init__(self, model_name, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(model_name,device=device)
        self.dimension=self.model.get_sentence_embedding_dimension()
    def encode(self,texts,batch_size=64):
        return self.model.encode(texts,batch_size=batch_size,convert_to_numpy=True,
            normalize_embeddings=True,show_progress_bar=False).tolist()
