from collections import defaultdict,deque
class ConversationMemory:
    def __init__(self,max_turns=8): self.sessions=defaultdict(lambda:deque(maxlen=max_turns))
    def add(self,session_id,user,assistant): self.sessions[session_id].append({"user":user,"assistant":assistant})
    def get(self,session_id): return list(self.sessions[session_id])
