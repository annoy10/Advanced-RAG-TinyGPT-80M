from collections import defaultdict
def reciprocal_rank_fusion(result_lists,k=60,limit=10):
    scores=defaultdict(float); docs={}
    for results in result_lists:
        for rank,doc in enumerate(results,1):
            did=doc["id"]; scores[did]+=1/(k+rank); docs[did]=doc
    return [dict(docs[i],rrf_score=scores[i]) for i in sorted(scores,key=scores.get,reverse=True)[:limit]]
class RAGFusion:
    def __init__(self,retriever,rewriter,rrf_k=60): self.retriever=retriever; self.rewriter=rewriter; self.rrf_k=rrf_k
    def search(self,query,k=10):
        lists=[self.retriever.search(q,k) for q in self.rewriter.rewrite(query)]
        return reciprocal_rank_fusion(lists,self.rrf_k,k)
