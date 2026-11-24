from pathlib import Path
import json
class IndexManager:
    def __init__(self,path="storage/bm25/index.json"): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def save(self,records): self.path.write_text(json.dumps(records,ensure_ascii=False),encoding="utf-8")
    def load(self): return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
