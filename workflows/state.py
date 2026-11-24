from dataclasses import dataclass,field
@dataclass
class RAGState:
    question:str
    retrieved:list=field(default_factory=list)
    answer:str=""
    citations:list=field(default_factory=list)
    errors:list=field(default_factory=list)
