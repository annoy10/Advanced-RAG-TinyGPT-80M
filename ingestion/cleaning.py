import re, unicodedata
def clean_text(text):
    text=unicodedata.normalize("NFKC", str(text)).replace("\x00"," ")
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()
