"""Offline smoke test for RAG tools without LLM tool selection.

Runs the three tools sequentially: ingest → search → qa with RAG_FAKE_QA=1.
"""

import asyncio
import os
try:
    from spoon_ai.tools.rag_tools import RAGIngestTool, RAGSearchTool, RAGQATool
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from spoon_ai.tools.rag_tools import RAGIngestTool, RAGSearchTool, RAGQATool


async def main():
    os.environ.setdefault("RAG_FAKE_QA", "1")
    ing = RAGIngestTool()
    sea = RAGSearchTool()
    qa = RAGQATool()

    print("[1] Ingest...")
    r1 = await ing.execute(inputs=["./doc"])  # small local docs
    print(r1)

    print("\n[2] Search...")
    r2 = await sea.execute(query="installation", top_k=3)
    print(r2.output[:400])

    print("\n[3] QA...")
    r3 = await qa.execute(question="How do I install the SDK?", top_k=3)
    print(r3.output[:400])
    print("Citations:", r3.system)


if __name__ == "__main__":
    asyncio.run(main())
