from workflows.rag_graph import RAGWorkflow
class FakeRetriever:
    def search(self,q,k):
        return [{"id":"x","text":"The capital of France is Paris.","parent_text":"The capital of France is Paris.",
                 "metadata":{"file_name":"facts.txt"},"fusion_score":0.9}]
class FakeGenerator:
    def answer(self,q,docs): return "The capital of France is Paris. [S1]"
def test_workflow_returns_cited_answer():
    state=RAGWorkflow(FakeRetriever(),FakeGenerator()).invoke("What is the capital of France?")
    assert "Paris" in state.answer
    assert "[S1]" in state.answer
