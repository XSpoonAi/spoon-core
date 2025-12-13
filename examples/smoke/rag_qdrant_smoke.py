"""Optional smoke test for Qdrant backend.

Requires:
- A running Qdrant at QDRANT_URL (default http://localhost:6333)
- pip install qdrant-client

Run:
  RAG_BACKEND=qdrant RAG_FAKE_QA=1 python examples/smoke/rag_qdrant_smoke.py
"""

import asyncio
import os
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
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from spoon_ai.rag import (
        get_default_config,
        get_vector_store,
        get_embedding_client,
        RagIndex,
        RagRetriever,
        RagQA,
    )


async def main():
    os.environ.setdefault("RAG_BACKEND", "qdrant")
    os.environ.setdefault("RAG_FAKE_QA", "1")

    try:
        import qdrant_client  # noqa: F401
    except Exception:
        print("[skip] qdrant-client not installed; skipping Qdrant smoke.")
        return

    cfg = get_default_config()
    store = get_vector_store(cfg.backend)
    embed = get_embedding_client(cfg.embeddings_provider)

    index = RagIndex(config=cfg, store=store, embeddings=embed)
    n = index.ingest(["./doc"])  # small local docs
    print(f"Ingested {n} chunks.")

    retr = RagRetriever(config=cfg, store=store, embeddings=embed)
    chunks = retr.retrieve("How to install?", top_k=3)
    qa = RagQA(config=cfg, llm=None)  # offline QA for smoke
    res = await qa.answer("How to install?", chunks)
    print("Answer:", res.answer[:200], "...")
    print("Citations:", res.citations)


if __name__ == "__main__":
    asyncio.run(main())
