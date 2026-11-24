def validate_chunks(chunks, min_chars=20, max_chars=4000):
    good=[]; rejected=[]
    for c in chunks:
        t=c.get("text","") if isinstance(c,dict) else str(c)
        if min_chars<=len(t.strip())<=max_chars: good.append(c)
        else: rejected.append(c)
    return good,rejected
