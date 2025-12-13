from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple
import math


class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        *,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        *,
        collection: str,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[List[Tuple[str, float, Dict]]]:
        """Return per-query list of (id, score, metadata). Higher score is better."""
        raise NotImplementedError

    @abstractmethod
    def delete_collection(self, collection: str) -> None:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        # storage: collection -> {id: (embedding, metadata)}
        self._data: Dict[str, Dict[str, Tuple[List[float], Dict]]] = {}

    def add(
        self,
        *,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ) -> None:
        if collection not in self._data:
            self._data[collection] = {}
        col = self._data[collection]
        for id_, vec, md in zip(ids, embeddings, metadatas):
            col[id_] = (vec, md)

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def query(
        self,
        *,
        collection: str,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[List[Tuple[str, float, Dict]]]:
        col = self._data.get(collection, {})
        results: List[List[Tuple[str, float, Dict]]] = []
        items = list(col.items())
        for q in query_embeddings:
            scored: List[Tuple[str, float, Dict]] = []
            for id_, (vec, md) in items:
                if filter:
                    # very simple exact-match filter on metadata keys
                    ok = True
                    for k, v in filter.items():
                        if md.get(k) != v:
                            ok = False
                            break
                    if not ok:
                        continue
                score = self._cosine(q, vec)
                scored.append((id_, score, md))
            scored.sort(key=lambda x: x[1], reverse=True)
            results.append(scored[: top_k])
        return results

    def delete_collection(self, collection: str) -> None:
        self._data.pop(collection, None)

