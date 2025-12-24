"""Smoke test: Qdrant (embedded) + Ollama embeddings.

Requirements:
- Ollama running locally (default: http://localhost:11434)
- An Ollama embedding model pulled locally (e.g. nomic-embed-text)
- pip install qdrant-client

Run (from core/):
  RAG_BACKEND=qdrant RAG_FAKE_QA=1 python examples/smoke/rag_qdrant_ollama_smoke.py

Notes:
- Uses embedded Qdrant via QDRANT_PATH=:memory: (no external service required).
- Auto-detects an embedding model from Ollama /api/tags if RAG_EMBEDDINGS_MODEL is not set.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

import requests

try:
    from spoon_ai.rag import (
        get_default_config,
        get_vector_store,
        get_embedding_client,
        RagIndex,
        RagRetriever,
        RagQA,
    )
except ModuleNotFoundError:
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from spoon_ai.rag import (
        get_default_config,
        get_vector_store,
        get_embedding_client,
        RagIndex,
        RagRetriever,
        RagQA,
    )

try:
    # Windows consoles can default to non-UTF8 encodings; avoid crashing on unicode output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


def _detect_ollama_embedding_model(base_url: str) -> Optional[str]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    models = []
    for m in data.get("models", []) or []:
        name = m.get("name") or m.get("model")
        if name:
            models.append(str(name))

    # Prefer obvious embedding models
    for name in models:
        lowered = name.lower()
        if "embed" in lowered or "embedding" in lowered:
            return name

    # Fallback to first model if present
    return models[0] if models else None


async def main() -> None:
    os.environ.setdefault("RAG_BACKEND", "qdrant")
    os.environ.setdefault("RAG_FAKE_QA", "1")

    # Force embedded/local Qdrant
    os.environ.setdefault("QDRANT_PATH", ":memory:")

    # Force Ollama embeddings
    os.environ.setdefault("RAG_EMBEDDINGS_PROVIDER", "ollama")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

    # Ensure qdrant-client is available
    try:
        import qdrant_client  # noqa: F401
    except Exception:
        print("[skip] qdrant-client not installed; install with: pip install qdrant-client")
        return

    provider = (os.getenv("RAG_EMBEDDINGS_PROVIDER") or "").strip().lower()
    if provider == "ollama":
        # Ensure Ollama is reachable
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            requests.get(f"{ollama_base.rstrip('/')}/api/tags", timeout=5).raise_for_status()
        except Exception as exc:
            print(f"[skip] Ollama not reachable at {ollama_base}: {exc}")
            return

        # Auto-detect embedding model if not set
        if not os.getenv("RAG_EMBEDDINGS_MODEL"):
            detected = _detect_ollama_embedding_model(ollama_base)
            if not detected:
                print("[skip] No Ollama models found. Pull an embedding model first, e.g.: ollama pull nomic-embed-text")
                return
            os.environ["RAG_EMBEDDINGS_MODEL"] = detected

    cfg = get_default_config()
    print(f"backend={cfg.backend} collection={cfg.collection}")
    print(f"embeddings_provider={cfg.embeddings_provider} embeddings_model={cfg.embeddings_model}")

    store = get_vector_store(cfg.backend)
    embed = get_embedding_client(cfg.embeddings_provider, model=cfg.embeddings_model)

    # Ingest a small local target (defaults to README.md on the core repo)
    index = RagIndex(config=cfg, store=store, embeddings=embed)
    docs_target = os.getenv("RAG_DOCS", "./README.md")
    n = index.ingest([docs_target])
    print(f"Ingested {n} chunks.")
    if n <= 0:
        print("[fail] No chunks ingested; check RAG_DOCS path or loader configuration.")
        return

    retr = RagRetriever(config=cfg, store=store, embeddings=embed)
    chunks = retr.retrieve("How to install?", top_k=3)
    print(f"Retrieved {len(chunks)} chunks.")

    # Offline QA (no LLM calls) - just to validate end-to-end flow
    qa = RagQA(config=cfg, llm=None)
    res = await qa.answer("How to install?", chunks)
    print("Answer preview:")
    print(res.answer[:400])


if __name__ == "__main__":
    asyncio.run(main())
