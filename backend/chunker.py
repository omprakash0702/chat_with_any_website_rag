import math

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    if not words:
        return []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        if end == n:
            break
        start = end - overlap
    return chunks
