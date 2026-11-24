from ingestion.document_parser import load_document
from ingestion.cleaning import clean_text
from ingestion.metadata_extractor import enrich_metadata
from ingestion.deduplication import deduplicate_documents
from chunking.parent_child_chunker import parent_child_chunks
from chunking.recursive_chunker import recursive_split
from chunking.semantic_chunker import semantic_chunk
from chunking.chunk_validator import validate_chunks
from config.settings import CHUNKING_STRATEGY

def prepare_sources(sources, strategy=None):
    strategy=(strategy or CHUNKING_STRATEGY).lower()
    docs=[]
    for source in sources:
        d=load_document(source); d["text"]=clean_text(d["text"]); docs.append(enrich_metadata(d))
    docs=deduplicate_documents(docs); records=[]
    for d in docs:
        if strategy=="parent_child":
            chunks=parent_child_chunks(d["text"])
        elif strategy=="semantic":
            chunks=[{"text":t,"parent_text":t,"parent_id":f"semantic_{i}","child_id":f"semantic_{i}"} for i,t in enumerate(semantic_chunk(d["text"]))]
        elif strategy=="recursive":
            chunks=[{"text":t,"parent_text":t,"parent_id":f"recursive_{i}","child_id":f"recursive_{i}"} for i,t in enumerate(recursive_split(d["text"]))]
        else:
            raise ValueError("CHUNKING_STRATEGY must be parent_child, semantic, or recursive")
        chunks,_=validate_chunks(chunks)
        for i,c in enumerate(chunks):
            records.append({"id":f'{d["content_hash"]}_{i}',"text":c["text"],"parent_text":c["parent_text"],
                "metadata":{**d["metadata"],"chunk_index":i,"parent_id":c["parent_id"]}})
    return records
