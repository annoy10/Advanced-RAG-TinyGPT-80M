def refuse_if_unsupported(docs):
    if not docs: return "The indexed sources do not contain enough relevant information to answer this question."
    return None
