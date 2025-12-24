"""Graph Agent demo using the RAG component (ingest -> retrieve -> answer).

Offline by default (hash embeddings + in-memory store).
"""

import asyncio
import os
from typing import Dict, Any, List

from spoon_ai.graph import StateGraph
from spoon_ai.agents.graph_agent import GraphAgent
from spoon_ai.chat import ChatBot

from spoon_ai.rag import (
    get_default_config,
    get_embedding_client,
    get_vector_store,
    RagIndex,
    RagRetriever,
    RagQA,
)


def build_pipeline():
    # Force offline backend: faiss or in-memory (not chroma)
    # Override config to ensure offline mode
    os.environ.setdefault("RAG_BACKEND", "faiss")
    os.environ.setdefault("RAG_EMBEDDINGS_PROVIDER", "hash")
    cfg = get_default_config()
    # Ensure we use faiss backend (fallback to in-memory if faiss not available)
    backend = "faiss" if cfg.backend != "faiss" else cfg.backend
    store = get_vector_store(backend)
    embed = get_embedding_client(
        cfg.embeddings_provider,
        model=cfg.embeddings_model,
    )

    index = RagIndex(config=cfg, store=store, embeddings=embed)
    retriever = RagRetriever(config=cfg, store=store, embeddings=embed)
    # Offline-friendly: if RAG_FAKE_QA=1, avoid initializing ChatBot
    llm = None if os.getenv("RAG_FAKE_QA") == "1" else ChatBot()
    qa = RagQA(config=cfg, llm=llm)

    def ingest_node(state: Dict[str, Any]) -> Dict[str, Any]:
        inputs = state.get("inputs", [])
        n = index.ingest(inputs)
        print(f"[ingest] inputs={inputs} chunks={n}")
        return {"ingested": n}

    def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        chunks = retriever.retrieve(q)
        return {"chunks": chunks, "context": retriever.build_context(chunks)}

    async def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        chunks = state.get("chunks", [])
        res = await qa.answer(q, chunks)
        return {"answer": res.answer, "citations": res.citations}

    graph = StateGraph(dict)
    graph.add_node("RAGIngestNode", ingest_node)
    graph.add_node("RAGRetrieveNode", retrieve_node)
    graph.add_node("RAGAnswerNode", answer_node)
    graph.add_edge("RAGIngestNode", "RAGRetrieveNode")
    graph.add_edge("RAGRetrieveNode", "RAGAnswerNode")
    graph.set_entry_point("RAGIngestNode")
    return graph.compile()


async def main():
    compiled = build_pipeline()
    # BaseAgent (parent of GraphAgent) requires an llm instance; use ChatBot by default
    agent = GraphAgent(name="rag-graph", graph=compiled.graph, llm=ChatBot())
    initial_state = {
        "inputs": ["./Readme.md"],  # small local doc
        "question": "How do I quickly install?",
    }
    result = await compiled.invoke(initial_state)
    print("\n== RAG Graph Agent Demo ==")
    print("Answer:", result.get("answer"))
    print("Citations:", result.get("citations"))


if __name__ == "__main__":
    asyncio.run(main())
