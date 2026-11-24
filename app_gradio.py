import gradio as gr
from chains.ingestion_chain import ingest_sources

def create_demo(workflow):
    def respond(message,history):
        if not message or not message.strip(): return "Please enter a question."
        try:
            state=workflow.invoke(message.strip())
            return state.answer
        except Exception as exc:
            return f"Request failed: {type(exc).__name__}: {exc}"
    def ingest(files):
        if not files: return "Choose at least one PDF, DOCX, or TXT file."
        paths=[f.name if hasattr(f,"name") else str(f) for f in files]
        try:
            # Reuse active workflow components through the supplied workflow.
            records=ingest_sources(paths)
            embedder=workflow.retriever.retriever.vector.embedder
            store=workflow.retriever.retriever.vector.store
            vectors=embedder.embed_documents([r["text"] for r in records])
            store.upsert(records,vectors)
            from vectorstore.index_manager import IndexManager
            idx=IndexManager(); old={r["id"]:r for r in idx.load()}
            old.update({r["id"]:r for r in records}); idx.save(list(old.values()))
            workflow.retriever.retriever.bm25=__import__("retrieval.bm25_retriever",fromlist=["BM25Retriever"]).BM25Retriever(idx.load())
            return f"Successfully indexed {len(records)} chunks from {len(paths)} file(s)."
        except Exception as exc: return f"Ingestion failed: {type(exc).__name__}: {exc}"
    with gr.Blocks(title="Advanced RAG | TinyGPT") as demo:
        gr.Markdown("# Advanced RAG — TinyGPT\nUpload your knowledge files, then ask questions grounded in those sources.")
        with gr.Tab("Chat"):
            gr.ChatInterface(fn=respond, chatbot=gr.Chatbot(height=520),
                textbox=gr.Textbox(placeholder="Ask a question about your indexed documents...",lines=2))
        with gr.Tab("Knowledge ingestion"):
            files=gr.File(file_count="multiple",file_types=[".pdf",".docx",".txt",".md"])
            button=gr.Button("Ingest documents",variant="primary")
            status=gr.Textbox(label="Ingestion status",interactive=False)
            button.click(ingest,inputs=files,outputs=status)
        gr.Markdown("**Grounding note:** Answers are extractive and source-grounded. The original TinyGPT checkpoint is retained but is not used as a general-purpose answer model because it was trained on sports QA.")
    return demo

if __name__=="__main__":
    from main import build_runtime
    create_demo(build_runtime()["workflow"]).launch()
