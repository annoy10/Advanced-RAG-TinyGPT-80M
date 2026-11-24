def test_embedding_module_import():
    from embeddings.batch_processor import batched
    assert list(batched([1,2,3],2))==[[1,2],[3]]
