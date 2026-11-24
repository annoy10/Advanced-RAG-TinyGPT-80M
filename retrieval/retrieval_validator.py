def validate_retrieval(docs,min_score=0.0):
    return [d for d in docs if d.get("score",d.get("fusion_score",0))>=min_score]
