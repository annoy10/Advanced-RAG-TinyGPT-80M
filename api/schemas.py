from pydantic import BaseModel, Field
class AskRequest(BaseModel):
    question:str=Field(min_length=1,max_length=4000)
class IngestRequest(BaseModel):
    paths:list[str]=Field(min_length=1)
class AskResponse(BaseModel):
    answer:str
    sources:list[dict]=[]
