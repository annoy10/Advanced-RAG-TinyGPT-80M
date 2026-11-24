class ChatHistory:
    def __init__(self,memory,session_id): self.memory=memory; self.session_id=session_id
    def add(self,user,assistant): self.memory.add(self.session_id,user,assistant)
    def messages(self): return self.memory.get(self.session_id)
