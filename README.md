# Advanced_RAG_TinyGPT_5060Ti

A modular, local-first Retrieval-Augmented Generation (RAG) application. It preserves the uploaded TinyGPT architecture and checkpoint, and adds document ingestion, chunking, dense + lexical retrieval, Reciprocal Rank Fusion, optional cross-encoder reranking, source citations, guardrails, Gradio UI, and a FastAPI service.

## Important model limitation

The supplied checkpoint was trained on the `leomaurodesenv/QASports2` sports-QA dataset. It is **not** a general instruction-following language model. It is preserved in `transformer/checkpoint/tinygpt.pt` and its architecture is in `transformer/`, but the default RAG answer path is extractive and evidence-grounded. This avoids pretending the small sports model can reliably synthesize answers about arbitrary uploaded documents. You can later connect a capable local instruct model through `generation/llm_manager.py`.

## Architecture

1. Ingestion: PDF, DOCX, TXT/MD, and web URL loaders; normalization, metadata, SHA-256 deduplication.
2. Chunking: recursive, sentence-oriented semantic-style, and parent-child chunking.
3. Embeddings: Sentence Transformers, normalized vectors, configurable device and batch encoding.
4. Storage: persistent local Qdrant for dense vectors; JSON BM25 corpus for lexical retrieval.
5. Retrieval: dense + BM25 hybrid search, query variants, RRF fusion.
6. Reranking: optional CrossEncoder; automatically falls back if the model cannot load.
7. Generation: extractive, context-grounded answer with source labels. No unsupported claims are intentionally generated.
8. Guardrails: relevance, basic faithfulness heuristic, unsupported-answer refusal, conflict metadata and confidence heuristic.
9. Interfaces: Gradio chat + upload UI and FastAPI routes.

## Windows / RTX 5060 Ti setup

Use Python 3.11 (recommended). Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build matching the installed NVIDIA driver from the official PyTorch selector: https://pytorch.org/get-started/locally/

Then:

```powershell
pip install -r requirements.txt
python scripts/benchmark_gpu.py
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Train the retained TinyGPT architecture again (optional; this downloads QASports2 and can take hours):

```powershell
python -m transformer.train_model
```

```text
```

If CUDA is unavailable, the app can run on CPU (slower). The `DEVICE` setting in `.env` can be set to `cpu` or `cuda`.

## Run the app

```powershell
python main.py
```

Open http://127.0.0.1:7860. Upload PDF/DOCX/TXT/MD documents in the **Knowledge ingestion** tab, wait for indexing to finish, then ask questions in **Chat**.

The first embedding/reranker use may download model weights from Hugging Face. For offline use, pre-download the models and set their local paths in `.env`.

## FastAPI

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs

Note: the API process must initialize and share a runtime before `/api/ask` and `/api/ingest` are available. For production, wire lifespan initialization and persistent job handling; do not expose this development server publicly as-is.

## Build an index from paths

```powershell
python scripts/build_index.py data/raw/guide.pdf data/raw/notes.docx
```

Supported inputs: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json`, and `http(s)://` URLs through the loader API. The Gradio upload UI currently accepts PDF, DOCX, TXT, and MD.

## Tests and code checks

```powershell
python -m compileall .
pytest -q
```

Tests are lightweight unit checks. Before production deployment, add a representative question/answer evaluation set and measure Recall@K, MRR, nDCG, answer correctness, citation precision, and faithfulness with human-reviewed labels.

## Data and privacy

Documents and the Qdrant database are stored locally under `data/` and `storage/`. Web ingestion makes an HTTP request to the provided URL. Embedding/reranker models may be downloaded from Hugging Face on first run. Review document permissions and remove sensitive files before sharing the project.

## Key settings

See `.env.example`. Important settings include embedding/reranker model names, chunk size/overlap, Qdrant collection, retrieval K values, and device.

## Known limitations / next production steps

- The semantic chunker currently uses sentence-boundary grouping, not embedding-boundary optimization.
- Faithfulness and confidence checks are heuristics, not calibrated factuality models.
- Conflict detection reports source distribution; it does not perform semantic contradiction resolution.
- BM25 corpus persistence is JSON-based and intended for a single local process. Use a dedicated search service or transactional store for concurrent production workloads.
- Parent text is stored with child points; context expansion is simple and can be tuned for token budgets.
- API runtime initialization and ingestion authentication need production hardening.
- The supplied `train_model_incomplete_original.py` is the uploaded source file retained for provenance; it ends mid-section and is not a runnable training script.

## Preserved uploaded files

The exact uploaded Gradio app and requirements file are retained in `legacy/`. The original TinyGPT model, transformer blocks, checkpoint, and incomplete training source are retained under `transformer/`.


## 80M TinyGPT upgrade

The 80M training configuration uses:
- Vocabulary: 18,079 (built from the QASports2 training corpus)
- Embedding dimension: 512
- Transformer layers: 20
- Attention heads: 8 (64 dimensions per head)
- Context/block size: 256
- Feed-forward width: 2,048
- Dropout: 0.1

With the current model implementation and vocabulary size, this configuration has approximately **81.71 million trainable parameters** (81,710,751 for the supplied 18,079-token vocabulary). The exact count is printed at training startup and saved in `transformer/checkpoint/tinygpt_80m.pt`.

The original 14M checkpoint remains as `transformer/checkpoint/tinygpt.pt`. The 80M checkpoint is a **new model with newly initialized weights** and must be trained; it is not an automatic enlargement of the old trained weights.

Train (downloads QASports2 and may take a long time):

```powershell
python -m transformer.train_model
```

After training finishes, inference automatically prefers `tinygpt_80m.pt`. If that file does not exist, it falls back to the original 14M checkpoint. Training uses batch size 16 and 8,000 steps by default; adjust these values in `transformer/train_model.py` if GPU memory is limited. Save the new checkpoint only after training completes.


## LangChain RAG integration

The application now uses LangChain LCEL for the RAG orchestration layer:
- `langchain_core.documents.Document` carries retrieved source passages and metadata.
- `RunnableParallel` runs the user question and hybrid retrieval branches.
- `RunnableLambda` composes retrieval, context/prompt preparation, and grounded response generation.
- The existing custom Qdrant + BM25 + hybrid retrieval + RRF + reranker stack is retained and exposed to the LangChain chain.
- `USE_LANGCHAIN_RAG=true` enables the LangChain chain (default).
- `GENERATION_BACKEND=extractive` is the safe default. It selects evidence-bearing sentences from retrieved chunks and cites sources.
- `GENERATION_BACKEND=tinygpt` enables the experimental TinyGPT generator. The supplied checkpoint is trained on sports QA, so it may not follow arbitrary document context reliably; verify its outputs carefully. It falls back to extractive generation if TinyGPT cannot load.

Install the added LangChain packages with:

```powershell
pip install -r requirements.txt
```

The exact three newly uploaded source files are retained in `transformer/original_uploads/`. The uploaded `train_model.py` ends immediately after the `TINYGPT MODEL` heading at line 366, so it is incomplete and is preserved as an original source snapshot only. The runnable 80M training implementation remains `transformer/train_model.py`; it does not overwrite the uploaded file.


## Power-loss-resilient 80M training (Windows)

The training script supports automatic resume:
- Saves an atomic recovery checkpoint to `transformer/checkpoint/tinygpt_80m_latest.pt` every 50 optimizer steps.
- A checkpoint includes model weights, AdamW optimizer state, AMP scaler, training step, tokenizer, and Python/NumPy/PyTorch/CUDA random-number-generator states.
- Writes to a temporary file, flushes it to disk, and then atomically replaces the previous recovery checkpoint. An interrupted write should leave the previous completed checkpoint intact.
- The tokenized dataset is cached locally as `transformer/checkpoint/qsports_token_cache.pt`, so it does not need to be tokenized again after a restart.
- After Windows restarts, run `.\scripts\resume_training.ps1` or `python -m transformer.train_model --resume auto`. Training resumes from the most recent saved update.
- `Ctrl+C` requests a graceful save after the current optimizer update.
- The final inference-compatible checkpoint is `transformer/checkpoint/tinygpt_80m.pt`. The original `tinygpt.pt` remains unchanged.

Example:

```powershell
python -m transformer.train_model --resume auto
```

Optional controls:

```powershell
python -m transformer.train_model --resume auto --steps 8000 --save-every 50 --batch-size 2 --grad-accum 8
```

For Windows desktops, set the plugged-in sleep timeout to Never while training (this does not protect against a power cut):

```powershell
powercfg /change standby-timeout-ac 0
```

**Important:** software cannot keep a desktop running when mains power is lost. It can only recover from the last successfully written checkpoint after power returns. A UPS is required if you want the PC to remain powered during an outage. Keep several GB of free disk space because resumable AdamW checkpoints include optimizer state and can be large.
