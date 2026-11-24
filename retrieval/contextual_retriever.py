class ContextualRetriever:
    def __init__(self,retriever): self.retriever=retriever
    def search(self,query,k=10): return self.retriever.search(query,k)
