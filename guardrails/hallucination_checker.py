from .faithfulness_checker import check_faithfulness
def check_hallucination(answer,context): return not check_faithfulness(answer,context)["faithful"]
