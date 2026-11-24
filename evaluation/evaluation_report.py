import json
from pathlib import Path
def save_report(metrics,path="data/evaluation/report.json"):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(metrics,indent=2))
