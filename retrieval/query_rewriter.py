import re
class QueryRewriter:
    def rewrite(self,query):
        q=" ".join(query.strip().split())
        variants=[q]
        # Conservative lexical variant; no external LLM assumptions.
        no_punct=re.sub(r"[^\w\s]"," ",q)
        no_punct=" ".join(no_punct.split())
        if no_punct.lower()!=q.lower(): variants.append(no_punct)
        return list(dict.fromkeys(variants))
