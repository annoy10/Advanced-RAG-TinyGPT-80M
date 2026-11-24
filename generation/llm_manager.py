from .answer_generator import AnswerGenerator
class LLMManager:
    def __init__(self): self.generator=AnswerGenerator()
    def answer(self,question,docs): return self.generator.generate(question,docs)
