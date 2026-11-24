def detect_conflicts(docs):
    # Flags materially different chunks from different sources for human review; does not infer truth.
    by_source={}
    for d in docs:
        src=d.get("metadata",{}).get("file_name","unknown")
        by_source.setdefault(src,[]).append(d.get("text",""))
    return [{"source":s,"chunks":len(v)} for s,v in by_source.items() if len(v)>1]
