from fastapi import FastAPI
from api.routes import router, runtime
from main import build_runtime
app=FastAPI(title="Advanced RAG TinyGPT API",version="1.0.0")
app.include_router(router,prefix="/api")
@app.on_event("startup")
def startup():
    runtime.update(build_runtime())
