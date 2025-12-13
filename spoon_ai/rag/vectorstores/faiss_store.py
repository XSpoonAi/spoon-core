from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .base import VectorStore


class FaissVectorStore(VectorStore):
    """FAISS-backed local vector store (cosine via inner product + L2 norm)."""

    def __init__(self) -> None:
        self._collections: Dict[str, Dict] = {}

    def _get_or_create(self, collection: str, dim: Optional[int] = None):
        import numpy as np  # noqa: F401
        import faiss  # type: ignore

        col = self._collections.get(collection)
        if col is None:
            if dim is None or dim <= 0:
                raise RuntimeError("FAISS collection not initialized and no dim provided")
            index = faiss.IndexFlatIP(dim)
            col = {
                "index": index,
                "ids": [],  # type: List[str]
                "metas": {},  # type: Dict[str, Dict]
                "dim": dim,
            }
            self._collections[collection] = col
        return col

    def _normalize(self, vecs: List[List[float]]):
        import numpy as np

        arr = np.asarray(vecs, dtype="float32")
        norms = (arr**2).sum(axis=1) ** 0.5
        norms[norms == 0] = 1.0
        arr = arr / norms[:, None]
        return arr

    def add(self, *, collection: str, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> None:
        import numpy as np

        if not embeddings:
            return
        dim = len(embeddings[0])
        col = self._get_or_create(collection, dim)
        if col["dim"] != dim:
            raise RuntimeError(f"FAISS dim mismatch: existing {col['dim']} vs new {dim}")

        arr = self._normalize(embeddings)
        col["index"].add(arr)
        col["ids"].extend(ids)
        for id_, md in zip(ids, metadatas):
            col["metas"][id_] = md

    def query(self, *, collection: str, query_embeddings: List[List[float]], top_k: int = 5, filter: Optional[Dict] = None) -> List[List[Tuple[str, float, Dict]]]:
        import numpy as np

        col = self._get_or_create(collection)
        if len(col["ids"]) == 0:
            return [[] for _ in query_embeddings]

        q = self._normalize(query_embeddings)
        scores, idxs = col["index"].search(q, min(top_k, len(col["ids"])))
        results: List[List[Tuple[str, float, Dict]]] = []
        for row_scores, row_idxs in zip(scores, idxs):
            triples: List[Tuple[str, float, Dict]] = []
            for s, i in zip(row_scores, row_idxs):
                if i < 0:
                    continue
                id_ = col["ids"][int(i)]
                md = col["metas"].get(id_, {})
                # simple metadata exact filter if provided
                if filter:
                    ok = True
                    for k, v in filter.items():
                        if md.get(k) != v:
                            ok = False
                            break
                    if not ok:
                        continue
                triples.append((id_, float(s), md))
            results.append(triples)
        return results

    def delete_collection(self, collection: str) -> None:
        self._collections.pop(collection, None)

