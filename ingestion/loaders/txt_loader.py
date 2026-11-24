from pathlib import Path
def load_txt(path):
    p=Path(path)
    return {"text": p.read_text(encoding="utf-8", errors="replace"), "source": str(p), "file_name": p.name, "file_type":"txt"}
