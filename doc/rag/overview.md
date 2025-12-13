RAG Overview
============

This package provides a minimal, switchable Retrieval-Augmented Generation (RAG) stack for SpoonAI:

- Unified APIs for indexing, retrieval, and QA with citations.
- Pluggable vector-store adapters via a registry and `RAG_BACKEND`.
- Embeddings via AnyRoute (OpenAI-compatible) or OpenAI; falls back to a deterministic offline hash embedding for tests/demos.
- Two runnable integrations: a ReAct Agent with tools, and a Graph Agent pipeline.

Key Components
- Embeddings: `AnyRouteEmbeddingClient`, `OpenAIEmbeddingClient`, `HashEmbeddingClient`.
- Vector Stores: `VectorStore` abstraction with a registry (`faiss|pinecone|qdrant|chroma` names map to a local in-memory store for offline use).
- Indexing: `RagIndex` (ingest local files/dirs, basic HTML/PDF parsing, URL fetch, chunking).
- Retrieval: `RagRetriever` (Top‑K with simple dedup and context assembly).
- QA: `RagQA` (prompt + citations via [n] markers, uses SpoonAI ChatBot/LLM stack).

Env Vars
- `RAG_BACKEND=faiss|pinecone|qdrant|chroma` (default: `faiss`)
- `RAG_COLLECTION` (default: `default`)
- `RAG_DIR` (default: `.rag_store`)
- `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- Embeddings/LLM:
  - AnyRoute: `ANYROUTE_API_KEY`, `ANYROUTE_BASE_URL`, `ANYROUTE_MODEL`
  - OpenAI: `OPENAI_API_KEY`

Notes
- Offline tests/examples default to the in-memory store and `hash` embeddings.
- Swap to real APIs by setting the appropriate keys; no code changes needed.

