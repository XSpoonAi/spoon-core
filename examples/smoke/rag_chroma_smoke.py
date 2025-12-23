"""Optional smoke test for Chroma backend.

Requires:
- pip install chromadb

Run:
  RAG_BACKEND=chroma RAG_FAKE_QA=1 python examples/smoke/rag_chroma_smoke.py
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
    os.environ.setdefault("RAG_BACKEND", "chroma")
    os.environ.setdefault("RAG_FAKE_QA", "1")

    try:
        import chromadb  # noqa: F401
    except Exception:
        print("[skip] chromadb not installed; skipping Chroma smoke.")
        return

    cfg = get_default_config()
    store = get_vector_store(cfg.backend)
    embed = get_embedding_client(cfg.embeddings_provider)

    index = RagIndex(config=cfg, store=store, embeddings=embed)
    n = index.ingest(["./README.md"])  # small local docs
    print(f"Ingested {n} chunks.")

    retr = RagRetriever(config=cfg, store=store, embeddings=embed)
    chunks = retr.retrieve("How do I configure API keys for SpoonOS?", top_k=3)
    qa = RagQA(config=cfg, llm=None)  # offline QA for smoke
    res = await qa.answer("How do I configure API keys for SpoonOS?", chunks)
    print("Answer:", res.answer[:200], "...")
    print("Citations:", res.citations)


if __name__ == "__main__":
    asyncio.run(main())
