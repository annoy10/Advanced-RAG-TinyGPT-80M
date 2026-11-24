from chunking.recursive_chunker import recursive_split
def test_chunk_limits():
    chunks=recursive_split("word "*2000,300,40)
    assert chunks and all(len(c)<=340 for c in chunks)
