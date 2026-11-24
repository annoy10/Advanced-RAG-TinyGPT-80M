import re
def format_citations(answer,docs):
    refs={}
    for i,d in enumerate(docs,1):
        m=d.get("metadata",{}); refs[f"S{i}"]=m.get("file_name") or m.get("source") or "unknown source"
    cited=sorted(set(re.findall(r"\[(S\d+)\]",answer)))
    if not cited: return answer
    lines=["\n\n**Sources**"]
    for c in cited:
        if c in refs: lines.append(f"- [{c}] {refs[c]}")
    return answer+"\n"+"\n".join(lines)
