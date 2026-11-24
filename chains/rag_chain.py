class RAGChain:
    def __init__(self,workflow): self.workflow=workflow
    def invoke(self,question): return self.workflow.invoke(question)
