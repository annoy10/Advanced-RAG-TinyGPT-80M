import logging
from pathlib import Path
from .settings import ROOT_DIR
def setup_logging():
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_dir/"app.log", encoding="utf-8"), logging.StreamHandler()])
    return logging.getLogger("advanced_rag")
