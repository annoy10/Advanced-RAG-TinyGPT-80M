class MultiQueryRetriever:
    def __init__(self,hybrid,rewriter): self.hybrid=hybrid; self.rewriter=rewriter
    def search(self,query,k=10):
        merged={}
        for q in self.rewriter.rewrite(query):
            for d in self.hybrid.search(q,k): merged[d["id"]]=d
        return sorted(merged.values(),key=lambda x:x.get("fusion_score",0),reverse=True)[:k]
