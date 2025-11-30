# backend/vectordb.py
import faiss
import numpy as np
import os
import json


class FaissStore:
    """
    Simple FAISS wrapper with in-memory metadata mapping.
    """
    def __init__(self, dim: int, index_path: str | None = None):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dim)  # using inner product (cosine if normalized)
        self.id_to_meta = {}
        self.next_id = 0

    def add(self, vectors: np.ndarray, metas: list[dict]):
        vectors = vectors.astype("float32")
        self.index.add(vectors)
        n = vectors.shape[0]
        for i in range(n):
            self.id_to_meta[self.next_id + i] = metas[i]
        self.next_id += n

    def search(self, qvec: np.ndarray, k: int = 5):
        qvec = qvec.astype("float32")
        D, I = self.index.search(qvec, k)
        results = []
        for row_scores, row_idxs in zip(D, I):
            for score, idx in zip(row_scores, row_idxs):
                if idx == -1:
                    continue
                meta = self.id_to_meta.get(int(idx))
                if meta is None:
                    continue
                results.append(
                    {
                        "score": float(score),
                        "meta": meta,
                    }
                )
        return results

    def save(self):
        if not self.index_path:
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path + ".index")
        with open(self.index_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.id_to_meta, f, ensure_ascii=False, indent=2)

    def load(self):
        if not self.index_path:
            return
        idx_path = self.index_path + ".index"
        meta_path = self.index_path + ".meta.json"
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            return
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.id_to_meta = json.load(f)
        # next_id = max key + 1
        if self.id_to_meta:
            self.next_id = max(map(int, self.id_to_meta.keys())) + 1
        else:
            self.next_id = 0
