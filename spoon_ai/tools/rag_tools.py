from __future__ import annotations

import os
from typing import Any, List, Optional

from pydantic import PrivateAttr

from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.chat import ChatBot
from spoon_ai.rag import (
    get_default_config,
    get_embedding_client,
    get_vector_store,
    RagIndex,
    RagRetriever,
    RagQA,
)


def _build_components():
    cfg = get_default_config()
    store = get_vector_store(cfg.backend)
    embed = get_embedding_client(
        cfg.embeddings_provider,
        model=cfg.embeddings_model,
    )
    return cfg, store, embed


class RAGIngestTool(BaseTool):
    name: str = "rag_ingest"
    description: str = "Ingest local files or URLs into the RAG index."
    parameters: dict = {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths or URLs to ingest",
            },
            "collection": {"type": "string", "description": "Optional collection name"},
        },
        "required": ["inputs"],
    }

    async def execute(self, *, inputs: List[str], collection: Optional[str] = None) -> ToolResult:
        cfg, store, embed = _build_components()
        index = RagIndex(config=cfg, store=store, embeddings=embed)
        n = index.ingest(inputs, collection=collection)
        return ToolResult(output=f"Ingested {n} chunks into collection '{collection or cfg.collection}'.")


class RAGSearchTool(BaseTool):
    name: str = "rag_search"
    description: str = "Search the RAG index and return top snippets."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "description": "Number of snippets to return (default: 5)"},
            "collection": {"type": "string", "description": "Collection to search (default: 'default')"},
        },
        "required": ["query"],
    }

    async def execute(self, *, query: str, top_k: Optional[int] = None, collection: Optional[str] = None) -> ToolResult:
        cfg, store, embed = _build_components()
        retr = RagRetriever(config=cfg, store=store, embeddings=embed)
        chunks = retr.retrieve(query, collection=collection, top_k=top_k)
        context = retr.build_context(chunks)
        return ToolResult(output=context)


class RAGQATool(BaseTool):
    name: str = "rag_qa"
    description: str = "Answer a question using the RAG index with citations."
    parameters: dict = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "top_k": {"type": "integer", "description": "Number of snippets to use (default: 5)"},
            "collection": {"type": "string", "description": "Collection to search (default: 'default')"},
        },
        "required": ["question"],
    }

    _llm: Optional[Any] = PrivateAttr(default=None)

    def __init__(self, llm: Optional[Any] = None, **data):
        super().__init__(**data)
        self._llm = llm

    async def execute(self, *, question: str, top_k: Optional[int] = None, collection: Optional[str] = None) -> ToolResult:
        cfg, store, embed = _build_components()
        retr = RagRetriever(config=cfg, store=store, embeddings=embed)
        chunks = retr.retrieve(question, collection=collection, top_k=top_k)

        # Use injected LLM if available, otherwise fallback (lazy init)
        # If RAG_FAKE_QA=1, avoid initializing ChatBot to prevent heavy deps
        if self._llm:
            llm = self._llm
        else:
            llm = None if os.getenv("RAG_FAKE_QA") == "1" else ChatBot()

        qa = RagQA(config=cfg, llm=llm)
        res = await qa.answer(question, chunks)
        return ToolResult(output=res.answer, system=str({"citations": res.citations}))


__all__ = ["RAGIngestTool", "RAGSearchTool", "RAGQATool"]
