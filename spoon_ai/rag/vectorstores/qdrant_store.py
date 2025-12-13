from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .base import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self, *, url: Optional[str] = None, api_key: Optional[str] = None, path: Optional[str] = None):
        self.url = url or os.getenv("QDRANT_URL") or "http://localhost:6333"
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.path = path or os.getenv("QDRANT_PATH")  # embedded/local mode if set
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except Exception as e:
                raise RuntimeError("qdrant-client package is required for Qdrant backend") from e
            if self.path or (self.url and self.url.startswith("file:")) or self.url == ":memory:":
                # embedded/local mode
                p = self.path or (self.url[5:] if self.url.startswith("file:") else ":memory:")
                self._client = QdrantClient(path=p)
            else:
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def _ensure_collection(self, name: str, dim: int):
        client = self._client_or_raise()
        from qdrant_client.http.models import Distance, VectorParams
        exists = False
        try:
            info = client.get_collection(name)
            exists = True
        except Exception:
            exists = False
        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, *, collection: str, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> None:
        client = self._client_or_raise()
        self._ensure_collection(collection, len(embeddings[0]) if embeddings else 1)
        points = []
        for id_, vec, md in zip(ids, embeddings, metadatas):
            points.append({"id": id_, "vector": vec, "payload": md})
        client.upsert(collection_name=collection, points=points)

    def query(self, *, collection: str, query_embeddings: List[List[float]], top_k: int = 5, filter: Optional[Dict] = None) -> List[List[Tuple[str, float, Dict]]]:
        client = self._client_or_raise()
        results: List[List[Tuple[str, float, Dict]]] = []
        for q in query_embeddings:
            res = client.search(collection_name=collection, query_vector=q, limit=top_k, with_payload=True)
            out: List[Tuple[str, float, Dict]] = []
            for r in res:
                out.append((str(r.id), float(r.score or 0.0), r.payload or {}))
            results.append(out)
        return results

    def delete_collection(self, collection: str) -> None:
        try:
            self._client_or_raise().delete_collection(collection)
        except Exception:
            pass
