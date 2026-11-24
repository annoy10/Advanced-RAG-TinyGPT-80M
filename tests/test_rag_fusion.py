from retrieval.rag_fusion import reciprocal_rank_fusion
def test_rrf_prefers_consensus():
    a=[{"id":"a","text":"A"},{"id":"b","text":"B"}]
    b=[{"id":"b","text":"B"},{"id":"a","text":"A"}]
    assert reciprocal_rank_fusion([a,b])[0]["id"] in {"a","b"}
