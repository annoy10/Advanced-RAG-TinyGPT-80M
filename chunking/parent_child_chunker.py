from .recursive_chunker import recursive_split
def parent_child_chunks(text, parent_size=2400, child_size=700, overlap=100):
    parents=recursive_split(text,parent_size,min(200,parent_size//4))
    out=[]
    for pi,parent in enumerate(parents):
        for ci,child in enumerate(recursive_split(parent,child_size,overlap)):
            out.append({"text":child,"parent_text":parent,"parent_id":f"p{pi}","child_id":f"p{pi}_c{ci}"})
    return out
