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
    chunk_size: int = 1200
    chunk_overlap: int = 120
    min_similarity: float = -10.0
    # Embeddings
    # - None/"auto": select an embedding-capable provider using core LLM config (env + fallback chain)
    # - "openai": force OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
    # - "openrouter": force OpenRouter embeddings (OpenAI-compatible /embeddings)
    # - "gemini": force Gemini embeddings (requires GEMINI_API_KEY + RAG_EMBEDDINGS_MODEL)
    # - "ollama": Ollama local embeddings (OLLAMA_BASE_URL + RAG_EMBEDDINGS_MODEL, auto-detects if not set)
    # - "openai_compatible": custom OpenAI-compatible embeddings endpoint
    #   Configure via: RAG_EMBEDDINGS_API_KEY + RAG_EMBEDDINGS_BASE_URL + RAG_EMBEDDINGS_MODEL
    # - "hash": deterministic offline fallback
    # Note: DeepSeek specializes in LLM (text generation), not embeddings.
    #       Use DeepSeek as LLM for QA generation, and other models for embeddings.
    embeddings_provider: Optional[str] = None
    embeddings_model: str = "text-embedding-3-small"  # Generic model name for all embedding providers
    # Reranking
    rerank_provider: Optional[str] = None
    rerank_model: Optional[str] = None
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
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
    min_similarity = float(os.getenv("RAG_MIN_SIMILARITY", "0.7"))
    embeddings_provider = os.getenv("RAG_EMBEDDINGS_PROVIDER")
    if embeddings_provider is not None:
        embeddings_provider = embeddings_provider.strip().lower() or None
    
    # Don't set a default model here. Each provider handles None/empty appropriately:
    # - OpenAI: uses OpenAIEmbeddingClient's default "text-embedding-3-small"
    # - Gemini: uses "models/embedding-001" (see embeddings.py line 339)
    # - OpenRouter: uses _derive_openrouter_embedding_model
    # - Ollama: raises error (requires explicit model)
    # - Hash: ignores model (not needed)
    embeddings_model = os.getenv("RAG_EMBEDDINGS_MODEL", "").strip()


    rerank_provider = os.getenv("RAG_RERANK_PROVIDER")
    if rerank_provider:
        rerank_provider = rerank_provider.strip().lower()
    rerank_model = os.getenv("RAG_RERANK_MODEL")

    return RagConfig(
        backend=backend,
        collection=collection,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_similarity=min_similarity,
        embeddings_provider=embeddings_provider,
        embeddings_model=embeddings_model,
        rerank_provider=rerank_provider,
        rerank_model=rerank_model,
        rag_dir=rag_dir,
    )
