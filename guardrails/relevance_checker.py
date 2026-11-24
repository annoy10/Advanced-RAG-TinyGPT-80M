import re
def relevance_score(query,text):
    q=set(re.findall(r"\w+",query.lower())); t=set(re.findall(r"\w+",text.lower()))
    return len(q&t)/max(1,len(q))
def is_relevant(query,docs,threshold=0.03):
    return any(relevance_score(query,d.get("text",""))>=threshold for d in docs)
