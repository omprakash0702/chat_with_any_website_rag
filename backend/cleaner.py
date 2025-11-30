from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

def extract_text_and_meta(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # remove scripts/styles
    for s in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
        s.decompose()

    title = (soup.title.string if soup.title else "").strip()
    # meta description
    desc = ""
    md = soup.find("meta", attrs={"name":"description"})
    if md and md.get("content"):
        desc = md["content"].strip()

    # main text heuristic: join <p> and headings
    blocks = []
    for tag in soup.find_all(["h1","h2","h3","p","li"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            blocks.append(text)

    text = "\n\n".join(blocks)
    text = re.sub(r'\n\s+\n', '\n\n', text)
    return {
        "url": url,
        "title": title,
        "description": desc,
        "text": text
    }
