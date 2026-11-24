def recall_at_k(retrieved_ids,relevant_ids,k):
    relevant=set(relevant_ids)
    return len(set(retrieved_ids[:k])&relevant)/max(1,len(relevant))
def reciprocal_rank(retrieved_ids,relevant_ids):
    relevant=set(relevant_ids)
    for i,x in enumerate(retrieved_ids,1):
        if x in relevant: return 1/i
    return 0.0
