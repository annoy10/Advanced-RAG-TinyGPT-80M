from .recursive_chunker import recursive_split
def semantic_chunk(text, chunk_size=900, overlap=120, embedder=None, similarity_threshold=0.45):
    # Sentence-boundary grouping; optional embeddings can be added without changing the API.
    import re
    sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text) if s.strip()]
    groups=[]; cur=""
    for sent in sentences:
        if cur and len(cur)+len(sent)+1>chunk_size:
            groups.append(cur); cur=sent
        else: cur=(cur+" "+sent).strip()
    if cur: groups.append(cur)
    result=[]
    for g in groups:
        result.extend(recursive_split(g,chunk_size,overlap))
    return result
