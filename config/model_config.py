from .settings import EMBEDDING_MODEL, RERANKER_MODEL, DEVICE
EMBEDDING_CONFIG = {"model_name": EMBEDDING_MODEL, "device": DEVICE, "normalize_embeddings": True}
RERANKER_CONFIG = {"model_name": RERANKER_MODEL, "device": DEVICE}
# Target TinyGPT architecture; the original checkpoint is retained separately.
TINY_GPT_CONFIG = {"block_size": 256, "embedding_dim": 512, "n_heads": 8, "n_layers": 20,
                   "dropout": 0.1, "target_parameters": 80_000_000,
                   "checkpoint": "transformer/checkpoint/tinygpt_80m.pt"}
