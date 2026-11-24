from urllib.parse import urlparse
def load_web(url, timeout=20):
    import requests
    from bs4 import BeautifulSoup
    r=requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    soup=BeautifulSoup(r.text,"html.parser")
    for x in soup(["script","style","nav","footer","header","noscript"]): x.decompose()
    title=soup.title.get_text(" ",strip=True) if soup.title else urlparse(url).netloc
    text=soup.get_text("\n",strip=True)
    return {"text":text, "source":url, "file_name":title, "file_type":"web"}
