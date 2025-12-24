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
        # list_indexes() shape varies across versions: can be list-like, dict, or object with attributes
        def _extract_index_names(obj) -> set:
            try:
                if obj is None:
                    return set()
                # Object with attribute 'indexes'
                if hasattr(obj, "indexes"):
                    seq = getattr(obj, "indexes")
                elif isinstance(obj, dict):
                    seq = obj.get("indexes") or obj.get("data") or obj.get("items") or obj
                else:
                    seq = obj
                names = set()
                if isinstance(seq, (list, tuple, set)):
                    for it in seq:
                        if isinstance(it, dict):
                            nm = it.get("name") or it.get("index_name")
                        else:
                            nm = getattr(it, "name", None)
                        if nm:
                            names.add(nm)
                elif isinstance(seq, dict):
                    # maybe {"name": ...}
                    nm = seq.get("name") or seq.get("index_name")
                    if nm:
                        names.add(nm)
                # Some SDKs expose `.names`
                if not names and hasattr(obj, "names"):
                    try:
                        names.update(set(getattr(obj, "names")))
                    except Exception:
                        pass
                return names
            except Exception:
                return set()

        try:
            indexes = _extract_index_names(pc.list_indexes())
        except Exception:
            indexes = set()
        if self.index_name not in indexes:
            if not dim:
                raise RuntimeError("Embedding dimension required to create Pinecone index on first use")
            # Use serverless spec defaults, overridable via env
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1")
            # Prefer new SDK with ServerlessSpec; only fall back if import fails
            try:
                from pinecone import ServerlessSpec
                spec = ServerlessSpec(cloud=cloud, region=region)
                try:
                    pc.create_index(
                        name=self.index_name,
                        dimension=dim,
                        metric="cosine",
                        spec=spec,
                    )
                except Exception as e:
                    # If index was created concurrently, ignore 409
                    try:
                        from pinecone.exceptions import PineconeApiException  # type: ignore
                    except Exception:
                        PineconeApiException = tuple()  # type: ignore
                    msg = str(e).lower()
                    status = getattr(e, "status", None)
                    if (hasattr(e, "__class__") and e.__class__.__name__ == "PineconeApiException") or isinstance(e, PineconeApiException):
                        if status == 409 or "already_exists" in msg or "already exists" in msg:
                            pass
                        else:
                            raise
                    else:
                        # Non-API exceptions should bubble up
                        raise
            except Exception as import_or_call_err:
                # If ServerlessSpec is missing (old SDK), try the older signature
                if "ServerlessSpec" in str(import_or_call_err):
                    try:
                        pc.create_index(
                            name=self.index_name,
                            dimension=dim,
                            metric="cosine",
                        )
                    except Exception as e:
                        # If the old signature still demands spec, re-raise with guidance
                        if "missing 1 required positional argument: 'spec'" in str(e):
                            raise RuntimeError(
                                "Your pinecone SDK requires ServerlessSpec. Please upgrade or set PINECONE_REGION/CLOUD."
                            ) from e
                        # Ignore 'already exists'
                        msg = str(e).lower()
                        status = getattr(e, "status", None)
                        if status == 409 or "already exists" in msg:
                            pass
                        else:
                            raise
                else:
                    # We only swallow import errors; other errors should surface
                    raise
        self._index = pc.Index(self.index_name)
        return self._index

    def add(self, *, collection: str, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> None:
        if not embeddings:
            return
        
        embedding_dim = len(embeddings[0])
        index = self._ensure_index(dim=embedding_dim)
        
        # Check if index dimension matches embedding dimension
        try:
            # Try to get index stats to check dimension
            stats = index.describe_index_stats()
            # Extract dimension from stats (format varies by SDK version)
            index_dim = None
            if isinstance(stats, dict):
                index_dim = stats.get("dimension") or stats.get("dim")
            elif hasattr(stats, "dimension"):
                index_dim = stats.dimension
            elif hasattr(stats, "dim"):
                index_dim = stats.dim
            
            if index_dim and index_dim != embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"Current embedding model produces {embedding_dim}-dimensional vectors, "
                    f"but Pinecone index '{self.index_name}' expects {index_dim}-dimensional vectors.\n"
                    f"Solutions:\n"
                    f"1. Delete the existing index and recreate it with the correct dimension:\n"
                    f"   - Use a different index name: RAG_PINECONE_INDEX=new-index-name\n"
                    f"   - Or delete the index manually in Pinecone console\n"
                    f"2. Use an embedding model that matches the index dimension ({index_dim}):\n"
                    f"   - For 1536 dimensions: text-embedding-3-large or text-embedding-ada-002\n"
                    f"   - For 768 dimensions: text-embedding-3-small\n"
                )
        except Exception as e:
            # If we can't check dimension, let Pinecone API handle it
            # but wrap the error with helpful message
            if "dimension" in str(e).lower() or "dim" in str(e).lower():
                raise ValueError(
                    f"Embedding dimension mismatch with Pinecone index '{self.index_name}'. "
                    f"Current embedding produces {embedding_dim}-dimensional vectors. "
                    f"Please use an embedding model that matches the index dimension, "
                    f"or delete the index and recreate it with the correct dimension."
                ) from e
            # Re-raise other exceptions as-is
            raise
        
        vectors = [
            {"id": id_, "values": [float(x) for x in vec], "metadata": md}
            for id_, vec, md in zip(ids, embeddings, metadatas)
        ]
        index.upsert(vectors=vectors, namespace=collection)

    def query(self, *, collection: str, query_embeddings: List[List[float]], top_k: int = 5, filter: Optional[Dict] = None) -> List[List[Tuple[str, float, Dict]]]:
        index = self._ensure_index()
        results: List[List[Tuple[str, float, Dict]]] = []
        for q in query_embeddings:
            # Pass filter dict directly (Pinecone uses Mongo-style filters)
            res = index.query(namespace=collection, vector=q, top_k=top_k, include_metadata=True, filter=filter)
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
