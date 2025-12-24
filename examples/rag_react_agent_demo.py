"""ReAct Agent demo using the RAG component (ingest -> search -> QA).

Runs offline by default using a deterministic hash embedding and in-memory store.
If OPENAI_API_KEY or OPENROUTER_API_KEY are present, will use those embeddings.

Env vars:
- RAG_BACKEND=faiss|pinecone|qdrant|chroma (default: faiss)
- RAG_COLLECTION=<name> (default: default)
- TOP_K, CHUNK_SIZE, CHUNK_OVERLAP optional
"""

import asyncio
import os
from spoon_ai.chat import ChatBot
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.tools.rag_tools import RAGIngestTool, RAGSearchTool, RAGQATool


async def main() -> None:
    tools = ToolManager([
        RAGIngestTool(),
        RAGSearchTool(),
        RAGQATool(),
    ])

    agent = ToolCallAgent(
        name="rag-react",
        llm=ChatBot(),
        available_tools=tools,
    )

    print("\n== RAG ReAct Agent Demo ==\n")

    # 1) Ingest a local directory or url
    docs_dir = os.getenv("RAG_DOCS", "./readme.md")
    ingest_request = f"Use rag_ingest to index docs in {docs_dir}"
    print("User:", ingest_request)
    out = await agent.run(ingest_request)
    print("Assistant:", out)

    # 2) Search
    search_request = "Use rag_search to find info about installation"
    print("\nUser:", search_request)
    out = await agent.run(search_request)
    print("Assistant:", out)

    # 3) QA
    qa_request = "Use rag_qa to answer: How do I install the SDK?"
    print("\nUser:", qa_request)
    out = await agent.run(qa_request)
    print("Assistant:", out)


if __name__ == "__main__":
    asyncio.run(main())

