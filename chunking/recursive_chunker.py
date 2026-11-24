from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
def recursive_split(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size<1 or overlap<0 or overlap>=chunk_size: raise ValueError("Require chunk_size > overlap >= 0")
    separators=["\n\n","\n",". ","? ","! ","; ",", "," "]
    def split(t):
        if len(t)<=chunk_size: return [t] if t.strip() else []
        for sep in separators:
            if sep in t:
                parts=t.split(sep); out=[]; cur=""
                for part in parts:
                    candidate=(cur+sep+part) if cur else part
                    if len(candidate)>chunk_size and cur:
                        out.extend(split(cur)); cur=part
                    else: cur=candidate
                if cur: out.extend(split(cur))
                if out: return out
        return [t[i:i+chunk_size] for i in range(0,len(t),chunk_size)]
    raw=split(text); chunks=[]; prev=""
    for item in raw:
        item=item.strip()
        if not item: continue
        if prev and len(prev)<overlap: prefix=prev
        else: prefix=prev[-overlap:] if overlap else ""
        merged=(prefix+" "+item).strip() if prefix and not item.startswith(prefix) else item
        chunks.append(merged[:chunk_size+overlap])
        prev=item
    return chunks
