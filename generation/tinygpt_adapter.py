class TinyGPTAdapter:
    """Optional original TinyGPT adapter. It is a sports-QA checkpoint, not a general instruction LLM."""
    def __init__(self):
        from transformer import model as tiny
        self.module=tiny
    def generate(self,prompt):
        return self.module.generate_answer(prompt,max_new_tokens=40,temperature=0.2)
