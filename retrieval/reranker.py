class Reranker:
    def __init__(self,model_name,device="cpu"):
        from sentence_transformers import CrossEncoder
        self.model=CrossEncoder(model_name,device=device)
    def rerank(self,query,docs,top_k=5):
        if not docs: return []
        pairs=[(query,d["text"]) for d in docs]
        scores=self.model.predict(pairs,show_progress_bar=False)
        ranked=[dict(d,rerank_score=float(s)) for d,s in zip(docs,scores)]
        return sorted(ranked,key=lambda x:x["rerank_score"],reverse=True)[:top_k]
