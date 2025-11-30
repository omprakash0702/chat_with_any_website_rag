# backend/indexer.py

import hashlib
import os
from urllib.parse import urldefrag
from backend.crawler import crawl
from backend.cleaner import extract_text_and_meta
from backend.chunker import chunk_text
from backend.embedder import embed_texts
from backend.vectordb import FaissStore


def page_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_site(root_url: str, max_pages: int = 10, index_path: str = "data/index/site"):
    """
    Default site indexer:
    - Crawls ONLY under this exact URL (no domain-wide wandering)
    - Cleans, chunks, embeds, indexes.
    """
    # strip #fragment (e.g. #Spin-offs)
    root_clean, _ = urldefrag(root_url)

    pages = crawl(
        root_clean,
        max_pages=max_pages,
        url_prefix=root_clean,  # 🔒 lock to this page / subtree
    )

    all_metas = []
    texts_for_bm25 = []

    for url, html in pages.items():
        meta_page = extract_text_and_meta(html, url)
        if not meta_page["text"].strip():
            continue

        chunks = chunk_text(meta_page["text"])
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "url": url,
                "title": meta_page["title"],
                "description": meta_page["description"],
                "chunk_id": f"{hashlib.md5((url + str(i)).encode()).hexdigest()}",
                "text": chunk,
            }
            all_metas.append(chunk_meta)
            texts_for_bm25.append(chunk)

    if not all_metas:
        store = FaissStore(dim=384, index_path=index_path)
        return store, texts_for_bm25, all_metas

    dim = None
    store = None
    batch_size = 64

    for start in range(0, len(all_metas), batch_size):
        batch_metas = all_metas[start:start + batch_size]
        batch_texts = [m["text"] for m in batch_metas]
        vecs = embed_texts(batch_texts, provider="local")

        if store is None:
            dim = vecs.shape[1]
            store = FaissStore(dim=dim, index_path=index_path)

        store.add(vecs, batch_metas)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    store.save()
    return store, texts_for_bm25, all_metas
