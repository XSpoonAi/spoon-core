import os
from dataclasses import dataclass
from typing import Optional

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    # Load .env immediately if available, but don't override existing env vars
    # This allows command-line env vars to override .env file values
    load_dotenv(override=False)
except ImportError:
    pass


@dataclass
class RagConfig:
    backend: str = "faiss"  # faiss|pinecone|qdrant|chroma
    collection: str = "default"
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    # Embeddings
    # - None/"auto": select an embedding-capable provider using core LLM config (env + fallback chain)
    # - "openai": force OpenAI embeddings
    # - "openrouter": force OpenRouter embeddings (OpenAI-compatible /embeddings)
    # - "gemini": force Gemini embeddings (requires GEMINI_API_KEY + RAG_EMBEDDINGS_MODEL)
    # - "ollama": Ollama local embeddings (OLLAMA_BASE_URL + RAG_EMBEDDINGS_MODEL)
    # - "openai_compatible": custom OpenAI-compatible embeddings (RAG_EMBEDDINGS_API_KEY + RAG_EMBEDDINGS_BASE_URL)
    # - "hash": deterministic offline fallback
    embeddings_provider: Optional[str] = None
    embeddings_model: str = "text-embedding-3-small"  # Generic model name for all embedding providers
    # Storage paths
    rag_dir: str = ".rag_store"
    
    @property
    def openai_embeddings_model(self) -> str:
        """Deprecated: use 'embeddings_model' instead. Kept for backward compatibility."""
        return self.embeddings_model

def get_default_config() -> RagConfig:
    backend = os.getenv("RAG_BACKEND", "faiss").lower()
    collection = os.getenv("RAG_COLLECTION", "default")
    rag_dir = os.getenv("RAG_DIR", ".rag_store")
    top_k = int(os.getenv("TOP_K", "5"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
    embeddings_provider = os.getenv("RAG_EMBEDDINGS_PROVIDER")
    if embeddings_provider is not None:
        embeddings_provider = embeddings_provider.strip().lower() or None
    embeddings_model = os.getenv("RAG_EMBEDDINGS_MODEL", "text-embedding-3-small").strip()

    return RagConfig(
        backend=backend,
        collection=collection,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings_provider=embeddings_provider,
        embeddings_model=embeddings_model,
        rag_dir=rag_dir,
    )
