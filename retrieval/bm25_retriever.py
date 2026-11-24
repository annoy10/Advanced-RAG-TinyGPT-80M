import re, math
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        """Small BM25 fallback for environments where rank-bm25 is not installed."""
        def __init__(self,corpus,k1=1.5,b=0.75):
            self.corpus=corpus; self.k1=k1; self.b=b
            self.avgdl=sum(map(len,corpus))/max(1,len(corpus))
            self.df={}
            for doc in corpus:
                for term in set(doc): self.df[term]=self.df.get(term,0)+1
        def get_scores(self,query):
            n=len(self.corpus); scores=[]
            for doc in self.corpus:
                freq={}
                for term in doc: freq[term]=freq.get(term,0)+1
                score=0.0
                for term in query:
                    f=freq.get(term,0)
                    if not f: continue
                    idf=math.log(1+(n-self.df.get(term,0)+0.5)/(self.df.get(term,0)+0.5))
                    score+=idf*(f*(self.k1+1))/(f+self.k1*(1-self.b+self.b*len(doc)/max(1,self.avgdl)))
                scores.append(score)
            return scores
class BM25Retriever:
    def __init__(self,records):
        self.records=records
        self.tokens=[self.tokenize(x["text"]) for x in records]
        self.index=BM25Okapi(self.tokens) if self.tokens else None
    def tokenize(self,text): return re.findall(r"\w+",text.lower())
    def search(self,query,k=10):
        if not self.index: return []
        scores=self.index.get_scores(self.tokenize(query))
        order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
        return [{"id":self.records[i]["id"],"text":self.records[i]["text"],"metadata":self.records[i].get("metadata",{}),
                 "score":float(scores[i]),"parent_text":self.records[i].get("parent_text","")} for i in order if scores[i]>0]
