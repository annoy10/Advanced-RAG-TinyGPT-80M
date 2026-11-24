class RetrievalChain:
    def __init__(self,retriever): self.retriever=retriever
    def invoke(self,query,k=5): return self.retriever.search(query,k)
