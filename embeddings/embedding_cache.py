import hashlib, json
from pathlib import Path
class EmbeddingCache:
    def __init__(self,path="storage/cache/embeddings.json"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {}
    def key(self,text): return hashlib.sha256(text.encode()).hexdigest()
    def get(self,text): return self.data.get(self.key(text))
    def set(self,text,vector):
        self.data[self.key(text)]=vector
        self.path.write_text(json.dumps(self.data))
