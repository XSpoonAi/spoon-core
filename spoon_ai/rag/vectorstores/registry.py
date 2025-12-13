import os
from typing import Optional

from .base import VectorStore, InMemoryVectorStore


def get_vector_store(backend: Optional[str] = None) -> VectorStore:
    """Return a vector store by backend name.

    Backends:
    - faiss: local/offline (mapped to in-memory cosine store)
    - pinecone: cloud Pinecone (requires PINECONE_API_KEY)
    - qdrant: local/cloud Qdrant (requires qdrant-client, default http://localhost:6333)
    - chroma: local Chroma (requires chromadb)
    """
    name = (backend or os.getenv("RAG_BACKEND", "faiss")).lower()
    if name == "pinecone":
        from .pinecone_store import PineconeVectorStore
        return PineconeVectorStore()
    if name == "qdrant":
        from .qdrant_store import QdrantVectorStore
        return QdrantVectorStore()
    if name == "chroma":
        from .chroma_store import ChromaVectorStore
        return ChromaVectorStore()
    if name == "faiss":
        try:
            import faiss  # noqa: F401
            from .faiss_store import FaissVectorStore
            return FaissVectorStore()
        except Exception:
            # fallback to in-memory if faiss not installed
            return InMemoryVectorStore()
    # Default: FAISS/local → InMemory
    return InMemoryVectorStore()
