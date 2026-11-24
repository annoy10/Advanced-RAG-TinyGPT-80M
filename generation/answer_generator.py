import re
def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text) if len(s.strip())>25]
class AnswerGenerator:
    """Reliable extractive baseline: returns evidence-bearing sentences rather than fabricating prose."""
    def generate(self,question,docs):
        if not docs: return "I could not find relevant information in the indexed documents."
        q=set(re.findall(r"\w+",question.lower()))
        ranked=[]
        for i,d in enumerate(docs,1):
            for sent in _sentences(d["text"]):
                toks=set(re.findall(r"\w+",sent.lower()))
                overlap=len(q&toks)/max(1,len(q))
                if overlap>0: ranked.append((overlap,sent,i))
        ranked.sort(reverse=True)
        selected=[]
        for score,sent,i in ranked:
            if sent not in [x[0] for x in selected]: selected.append((sent,i))
            if len(selected)>=4: break
        if not selected:
            d=docs[0]; return f"The retrieved document contains related material, but I cannot verify a direct answer from the available text. [S1]"
        return " ".join(f"{s} [S{i}]" for s,i in selected)
