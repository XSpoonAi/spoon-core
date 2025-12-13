from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .base import VectorStore


class PineconeVectorStore(VectorStore):
    def __init__(self, *, api_key: Optional[str] = None, index_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise RuntimeError("PINECONE_API_KEY is required for Pinecone backend")
        self.index_name = index_name or os.getenv("RAG_PINECONE_INDEX", "spoon-rag")
        self._pc = None
        self._index = None

    def _ensure_client(self):
        if self._pc is None:
            try:
                from pinecone import Pinecone
            except Exception as e:
                raise RuntimeError("pinecone package is required for Pinecone backend") from e
            self._pc = Pinecone(api_key=self.api_key)
        return self._pc

    def _ensure_index(self, dim: Optional[int] = None):
        pc = self._ensure_client()
        if self._index is not None:
            return self._index
        # Create index if missing (serverless default)
        indexes = {idx["name"] for idx in pc.list_indexes()}
        if self.index_name not in indexes:
            if not dim:
                raise RuntimeError("Embedding dimension required to create Pinecone index on first use")
            pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
            )
        self._index = pc.Index(self.index_name)
        return self._index

    def add(self, *, collection: str, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> None:
        index = self._ensure_index(dim=len(embeddings[0]) if embeddings else None)
        vectors = [
            {"id": id_, "values": vec, "metadata": md}
            for id_, vec, md in zip(ids, embeddings, metadatas)
        ]
        index.upsert(vectors=vectors, namespace=collection)

    def query(self, *, collection: str, query_embeddings: List[List[float]], top_k: int = 5, filter: Optional[Dict] = None) -> List[List[Tuple[str, float, Dict]]]:
        index = self._ensure_index()
        results: List[List[Tuple[str, float, Dict]]] = []
        for q in query_embeddings:
            res = index.query(namespace=collection, vector=q, top_k=top_k, include_metadata=True)
            matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
            out: List[Tuple[str, float, Dict]] = []
            for m in matches:
                mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
                sc = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
                md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
                out.append((mid, float(sc or 0.0), md or {}))
            results.append(out)
        return results

    def delete_collection(self, collection: str) -> None:
        index = self._ensure_index()
        # Delete all vectors in namespace
        try:
            index.delete(namespace=collection, delete_all=True)
        except Exception:
            pass

