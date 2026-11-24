import uuid
from qdrant_client import QdrantClient, models
class QdrantStore:
    def __init__(self,path,collection,vector_size):
        self.client=QdrantClient(path=str(path)); self.collection=collection; self.vector_size=vector_size
        names=[c.name for c in self.client.get_collections().collections]
        if collection not in names:
            self.client.create_collection(collection_name=collection,
                vectors_config=models.VectorParams(size=vector_size,distance=models.Distance.COSINE))
    def upsert(self,records,vectors):
        points=[]
        for r,v in zip(records,vectors):
            payload={k:x for k,x in r.items() if k!="vector"}
            points.append(models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,r["id"])),vector=v,payload=payload))
        if points: self.client.upsert(collection_name=self.collection,points=points,wait=True)
    def search(self,vector,limit=10,filters=None):
        qfilter=None
        if filters:
            qfilter=models.Filter(must=[models.FieldCondition(key=k,match=models.MatchValue(value=v)) for k,v in filters.items()])
        hits=self.client.query_points(collection_name=self.collection,query=vector,limit=limit,query_filter=qfilter,with_payload=True).points
        return [{"id":h.payload.get("id"),"text":h.payload.get("text",""),"metadata":h.payload.get("metadata",{}),
                 "score":float(h.score),"parent_text":h.payload.get("parent_text","")} for h in hits]
    def count(self):
        return self.client.count(collection_name=self.collection,exact=True).count
