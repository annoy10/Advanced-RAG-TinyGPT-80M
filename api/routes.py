from fastapi import APIRouter, HTTPException
from api.schemas import AskRequest, IngestRequest
router=APIRouter()
runtime={}
@router.post("/ask")
def ask(payload:AskRequest):
    if runtime.get("workflow") is None: raise HTTPException(503,"RAG index is not ready. Ingest documents first.")
    state=runtime["workflow"].invoke(payload.question)
    return {"answer":state.answer,"sources":state.citations}
@router.post("/ingest")
def ingest(payload:IngestRequest):
    try:
        records=runtime["ingest"](payload.paths)
        runtime["index"](records)
        return {"ingested_chunks":len(records)}
    except Exception as e: raise HTTPException(400,str(e))
@router.get("/health")
def health(): return {"status":"ok","indexed":runtime.get("workflow") is not None}
