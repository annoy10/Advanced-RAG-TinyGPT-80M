import hashlib
def content_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
def deduplicate_documents(docs):
    seen=set(); out=[]
    for d in docs:
        h=content_hash(d.get("text",""))
        if h and h not in seen:
            seen.add(h); d["content_hash"]=h; out.append(d)
    return out
