from pathlib import Path
from datetime import datetime, timezone
def enrich_metadata(doc):
    src=doc.get("source","")
    doc.setdefault("metadata",{})
    doc["metadata"].update({"source":src,"file_name":doc.get("file_name",Path(src).name),
        "file_type":doc.get("file_type","unknown"),"ingested_at":datetime.now(timezone.utc).isoformat()})
    return doc
