def confidence_from_scores(docs):
    if not docs: return {"level":"low","score":0.0}
    vals=[float(d.get("rerank_score",d.get("fusion_score",d.get("score",0)))) for d in docs]
    top=max(vals)
    # Relative heuristic only; not a calibrated probability.
    level="high" if top>0.5 else "medium" if top>0.1 else "low"
    return {"level":level,"score":top,"is_calibrated":False}
