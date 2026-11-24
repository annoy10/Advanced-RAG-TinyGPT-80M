from collections import defaultdict
class HybridRetriever:
    def __init__(self,vector_retriever,bm25_retriever,rrf_k=60): self.vector=vector_retriever; self.bm25=bm25_retriever; self.rrf_k=rrf_k
    def search(self,query,k=10):
        a=self.vector.search(query,k); b=self.bm25.search(query,k); scores=defaultdict(float); docs={}
        for results in (a,b):
            for rank,d in enumerate(results,1):
                key=d["id"]; scores[key]+=1/(self.rrf_k+rank); docs[key]=d
        return [dict(docs[x],fusion_score=scores[x]) for x in sorted(scores,key=scores.get,reverse=True)[:k]]
