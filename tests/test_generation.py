from generation.answer_generator import AnswerGenerator
def test_generator_uses_context():
    d=[{"id":"1","text":"The capital of France is Paris.","metadata":{"file_name":"x.txt"}}]
    assert "Paris" in AnswerGenerator().generate("What is the capital of France?",d)
