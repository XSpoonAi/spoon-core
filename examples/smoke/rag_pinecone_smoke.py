"""Optional smoke test for Pinecone backend.

Requires:
- PINECONE_API_KEY
Optionally:
- RAG_PINECONE_INDEX (default: spoon-rag)

Run:
  RAG_BACKEND=pinecone RAG_FAKE_QA=1 python examples/smoke/rag_pinecone_smoke.py
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
    if not os.getenv("PINECONE_API_KEY"):
        print("[skip] Missing PINECONE_API_KEY; skipping Pinecone smoke.")
        return

    os.environ.setdefault("RAG_BACKEND", "pinecone")
    os.environ.setdefault("RAG_FAKE_QA", "1")

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
