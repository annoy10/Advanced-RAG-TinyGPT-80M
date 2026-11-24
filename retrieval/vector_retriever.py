class VectorRetriever:
    def __init__(self,store,embedder): self.store=store; self.embedder=embedder
    def search(self,query,k=10,filters=None): return self.store.search(self.embedder.embed_query(query),k,filters)
