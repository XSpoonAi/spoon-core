"""Smoke test: ingest a GitHub blob URL (auto-convert to raw).

It validates that the loader converts GitHub web URLs:
  https://github.com/<org>/<repo>/blob/<ref>/<path>
into raw content URLs:
  https://raw.githubusercontent.com/<org>/<repo>/<ref>/<path>

Run (offline QA):
  RAG_BACKEND=faiss RAG_FAKE_QA=1 python examples/smoke/rag_github_loader_test.py

Optionally, you can enable real QA by setting RAG_FAKE_QA=0 and configuring your LLM env.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys

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

from spoon_ai.chat import ChatBot

TEST_URL = "https://github.com/XSpoonAi/spoon-core/blob/main/README.md"
DB_DIR = ".rag_test_github"

try:
    # Windows consoles can default to non-UTF8 encodings; avoid crashing on unicode output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


async def main() -> None:
    os.environ.setdefault("RAG_BACKEND", "faiss")
    os.environ.setdefault("RAG_FAKE_QA", "1")
    os.environ.setdefault("RAG_DIR", DB_DIR)
    os.environ.setdefault("CHUNK_SIZE", "1000")
    os.environ.setdefault("CHUNK_OVERLAP", "50")

    # Cleanup previous runs
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR, ignore_errors=True)

    cfg = get_default_config()
    store = get_vector_store(cfg.backend)
    embed = get_embedding_client(cfg.embeddings_provider, model=cfg.embeddings_model)

    print("== RAG GitHub Loader Smoke ==")
    print("URL:", TEST_URL)

    index = RagIndex(config=cfg, store=store, embeddings=embed)
    count = index.ingest([TEST_URL])
    print("Ingested chunks:", count)

    if count <= 0:
        print("[fail] No chunks ingested.")
        return

    retriever = RagRetriever(config=cfg, store=store, embeddings=embed)
    chunks = retriever.retrieve("What is Spoon-Core?", top_k=3)
    print("Retrieved:", len(chunks))

    # Quick sanity: if we ingested HTML UI, we'd likely see <html> / <body> etc.
    joined = "\n".join([c.text for c in chunks])
    if "<html" in joined.lower() or "<body" in joined.lower():
        print("[warn] Retrieved content looks like HTML; GitHub blob->raw conversion may not have applied.")
    else:
        print("[ok] Retrieved content looks like raw markdown/text.")

    llm = None if os.getenv("RAG_FAKE_QA") == "1" else ChatBot()
    qa = RagQA(config=cfg, llm=llm)
    res = await qa.answer("What is Spoon-Core?", chunks)

    print("Answer preview:")
    print(res.answer[:500])

    # Cleanup
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
