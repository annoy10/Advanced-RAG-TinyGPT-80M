from retrieval.bm25_retriever import BM25Retriever
def test_bm25_returns_matching_doc():
    r=BM25Retriever([{"id":"1","text":"football striker scores goal","metadata":{}},{"id":"2","text":"cooking recipe"}])
    assert r.search("football goal",1)[0]["id"]=="1"
