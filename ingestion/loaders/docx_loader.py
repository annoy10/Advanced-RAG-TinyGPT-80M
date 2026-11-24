from pathlib import Path
def load_docx(path):
    try:
        from docx import Document
    except ImportError as e:
        raise RuntimeError("Install python-docx: pip install python-docx") from e
    p=Path(path); d=Document(p)
    parts=[x.text for x in d.paragraphs if x.text.strip()]
    for table in d.tables:
        for row in table.rows: parts.append(" | ".join(c.text.strip() for c in row.cells))
    return {"text":"\n".join(parts), "source":str(p), "file_name":p.name, "file_type":"docx"}
