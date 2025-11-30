# backend/embedder.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

_MODEL_CACHE = {}
_OPENAI_CLIENT = None


def get_local_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load / cache the local sentence-transformers model."""
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def get_openai_client():
    """Return OpenAI client if API key is set, else None."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not OPENAI_AVAILABLE:
        return None

    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def embed_texts(texts, provider: str = "local"):
    """
    Embed a list of texts.

    provider:
      - "local"  -> sentence-transformers (no API key needed)
      - "openai" -> OpenAI embeddings (if OPENAI_API_KEY set)
      - "auto"   -> try OpenAI, fall back to local
    """
    if provider == "local":
        model = get_local_model()
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs.astype("float32")

    if provider in ("openai", "auto"):
        client = get_openai_client()
        if client is None and provider == "openai":
            raise RuntimeError("OPENAI_API_KEY not set or openai package unavailable.")
        if client is not None:
            # Small, simple batching; can be improved later.
            res = client.embeddings.create(
                input=texts,
                model="text-embedding-3-large",
            )
            embs = [item.embedding for item in res.data]
            return np.array(embs, dtype="float32")

    # Fallback
    model = get_local_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs.astype("float32")
