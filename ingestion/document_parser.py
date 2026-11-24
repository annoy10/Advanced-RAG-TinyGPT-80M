from pathlib import Path
from .loaders.pdf_loader import load_pdf
from .loaders.docx_loader import load_docx
from .loaders.txt_loader import load_txt
from .loaders.web_loader import load_web
def load_document(source):
    s=str(source)
    if s.startswith(("http://","https://")): return load_web(s)
    ext=Path(s).suffix.lower()
    if ext==".pdf": return load_pdf(s)
    if ext==".docx": return load_docx(s)
    if ext in {".txt",".md",".csv",".json"}: return load_txt(s)
    raise ValueError(f"Unsupported file type: {ext}")
