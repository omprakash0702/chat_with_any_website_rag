# backend/retriever.py
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    """
    Hybrid retriever combining BM25 (text-based) and dense vector similarity.
    - faiss_store: FaissStore
    - texts: list[str] (aligned with metas)
    - metas: list[dict] (same index as texts)
    """
    def __init__(self, faiss_store, texts_for_bm25, metas):
        self.faiss = faiss_store
        self.texts = texts_for_bm25
        self.metas = metas

        tokenized = [t.split() for t in self.texts]
        if len(tokenized) == 0:
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, qvec, top_k: int = 5, mix_weight: float = 0.5):
        """
        Returns list of:
          {"meta": {...}, "score": float}
        """
        candidates = {}

        # BM25 branch
        if self.bm25 is not None and self.texts:
            bm25_scores = self.bm25.get_scores(query.split())
            bm25_idxs = np.argsort(bm25_scores)[::-1][:top_k]

            for idx in bm25_idxs:
                score = float(bm25_scores[idx])
                meta = self.metas[idx]
                key = meta["chunk_id"]
                # small bonus if query appears in title
                title = (meta.get("title") or "").lower()
                if any(w.lower() in title for w in query.split()):
                    score *= 1.2
                prev = candidates.get(key, {"meta": meta, "score": 0.0})
                prev["score"] += mix_weight * score
                candidates[key] = prev

        # Dense branch
        dense_results = self.faiss.search(qvec, k=top_k)
        for r in dense_results:
            meta = r["meta"]
            score = r["score"]
            key = meta["chunk_id"]
            prev = candidates.get(key, {"meta": meta, "score": 0.0})
            prev["score"] += (1.0 - mix_weight) * score
            candidates[key] = prev

        # Sort and slice
        results = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]
