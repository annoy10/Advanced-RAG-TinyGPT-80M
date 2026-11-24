def build_context(docs,max_chars=12000):
    parts=[]; used=0
    for i,d in enumerate(docs,1):
        meta=d.get("metadata",{}); source=meta.get("file_name") or meta.get("source") or "unknown"
        label=f"S{i}"
        text=d.get("parent_text") or d["text"]
        block=f"[{label}] Source: {source}\n{text}"
        if used+len(block)>max_chars: break
        parts.append(block); used+=len(block)
    return "\n\n".join(parts)
