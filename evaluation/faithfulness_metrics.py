from guardrails.faithfulness_checker import check_faithfulness
def faithfulness_score(answer,context):
    result=check_faithfulness(answer,context)
    return {"score":float(result["faithful"]),"checked_claims":result["checked_claims"]}
