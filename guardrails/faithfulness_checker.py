import re
def supported_sentence(sentence,context,threshold=0.12):
    a=set(re.findall(r"\w+",sentence.lower())); b=set(re.findall(r"\w+",context.lower()))
    return len(a&b)/max(1,len(a))>=threshold
def check_faithfulness(answer,context):
    claims=[x.strip() for x in re.split(r"(?<=[.!?])\s+",answer) if x.strip()]
    return {"faithful":all(supported_sentence(re.sub(r"\[S\d+\]","",c),context) for c in claims),
            "checked_claims":len(claims)}
