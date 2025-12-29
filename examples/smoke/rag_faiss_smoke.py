"""Optional smoke test for FAISS backend.

If faiss is not installed, falls back to in-memory store.

Run:
  RAG_BACKEND=faiss RAG_FAKE_QA=1 python examples/smoke/rag_faiss_smoke.py
"""

import asyncio
import os
from spoon_ai.chat import ChatBot
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
    os.environ.setdefault("RAG_BACKEND", "faiss")
    #os.environ.setdefault("RAG_FAKE_QA", "1")

    cfg = get_default_config()
    store = get_vector_store(cfg.backend)
    embed = get_embedding_client(cfg.embeddings_provider)

    index = RagIndex(config=cfg, store=store, embeddings=embed)
    n = index.ingest(["./tests"])  # small local docs
    print(f"Ingested {n} chunks.")
    query = "What is the code for the add_message_with_image method in sooncore，If the answer is not in the context, say you don't know"
    retr = RagRetriever(config=cfg, store=store, embeddings=embed)
    chunks = retr.retrieve(query, top_k=3)

    print(f"\n[Debug] Retrieved {len(chunks)} chunks for query: {query}")
    for i, c in enumerate(chunks, 1):
        src_name = os.path.basename(c.metadata.get("source", "unknown"))
        print(f"  [{i}] Score: {c.score:.4f}")
        print(f"      Source: {src_name}")
        content_preview = c.text#[:200].replace("\n", "\\n")
        print(f"      Content: {content_preview}...")
        print("-" * 40)
    llm_client = ChatBot(llm_provider="openrouter")
    qa = RagQA(config=cfg, llm=llm_client)  # offline QA for smoke
    res = await qa.answer(query, chunks)
    print("Answer:", res.answer[:200], "...")
    print("Citations:", res.citations)


if __name__ == "__main__":
    asyncio.run(main())
