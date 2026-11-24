from pathlib import Path
def load_pdf(path):
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("Install PyMuPDF: pip install pymupdf") from e
    p=Path(path); doc=fitz.open(p); pages=[]
    for i,page in enumerate(doc):
        t=page.get_text("text")
        if t.strip(): pages.append(f"[Page {i+1}]\n{t}")
    return {"text":"\n\n".join(pages), "source":str(p), "file_name":p.name, "file_type":"pdf"}
