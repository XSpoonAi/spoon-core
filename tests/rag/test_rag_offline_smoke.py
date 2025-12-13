import asyncio
import os

from spoon_ai.rag.embeddings import HashEmbeddingClient
from spoon_ai.rag.vectorstores.base import InMemoryVectorStore
from spoon_ai.rag.config import RagConfig
from spoon_ai.rag.index import RagIndex
from spoon_ai.rag.retriever import RagRetriever
from spoon_ai.rag.qa import RagQA


def test_offline_qa_via_env(tmp_path, monkeypatch):
    # Force offline QA
    monkeypatch.setenv("RAG_FAKE_QA", "1")

    f = tmp_path / "doc.txt"
    f.write_text("Install by running: pip install spoon-ai-sdk. Then enjoy.")

    cfg = RagConfig(
        backend="faiss",
        collection="qa-offline",
        top_k=2,
        chunk_size=200,
        chunk_overlap=0,
        embeddings_provider="hash",
    )
    store = InMemoryVectorStore()
    embed = HashEmbeddingClient(dim=64)
    index = RagIndex(config=cfg, store=store, embeddings=embed)
    assert index.ingest([str(f)]) > 0

    retriever = RagRetriever(config=cfg, store=store, embeddings=embed)
    chunks = retriever.retrieve("How to install the SDK?", top_k=2)
    qa = RagQA(config=cfg, llm=None)

    async def _run():
        res = await qa.answer("How to install the SDK?", chunks)
        assert isinstance(res.answer, str)
        assert isinstance(res.citations, list)
        assert len(res.citations) >= 1

    asyncio.run(_run())


class _StringChatBot:
    async def ask(self, messages, system_msg=None, output_queue=None):
        # Return a plain string to ensure RagQA accepts it
        return "Install with pip. [1]"


def test_llm_returns_str_is_accepted(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Install with pip. SpoonAI makes it simple.")

    cfg = RagConfig(
        backend="faiss",
        collection="qa-str",
        top_k=1,
        chunk_size=128,
        chunk_overlap=0,
        embeddings_provider="hash",
    )
    store = InMemoryVectorStore()
    embed = HashEmbeddingClient(dim=64)
    index = RagIndex(config=cfg, store=store, embeddings=embed)
    assert index.ingest([str(f)]) > 0

    retriever = RagRetriever(config=cfg, store=store, embeddings=embed)
    chunks = retriever.retrieve("install", top_k=1)
    qa = RagQA(config=cfg, llm=_StringChatBot())

    async def _run():
        res = await qa.answer("install", chunks)
        assert isinstance(res.answer, str) and "pip" in res.answer.lower()

    asyncio.run(_run())

