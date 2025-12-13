from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Any

import os
if TYPE_CHECKING:  # Avoid heavy imports unless actually used by LLM path
    from spoon_ai.chat import ChatBot, Message  # type: ignore

from .config import RagConfig
from .retriever import RetrievedChunk


@dataclass
class QAResult:
    answer: str
    citations: List[Dict]


DEFAULT_QA_SYSTEM = (
    "You are a helpful assistant that answers using the provided context. "
    "Always cite sources using [n] markers that refer to the numbered context snippets."
)

QA_PROMPT_TEMPLATE = (
    "Answer the user question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "- Use [n] markers in the answer to cite the snippet numbers.\n"
    "- If the answer is unknown, say you don't know.\n"
)


class RagQA:
    def __init__(self, *, config: RagConfig, llm: Any):
        self.config = config
        self.llm = llm

    async def answer(self, question: str, chunks: List[RetrievedChunk]) -> QAResult:
        # Optional offline fallback: set RAG_FAKE_QA=1 to synthesize an answer locally
        if os.getenv("RAG_FAKE_QA") == "1" or not hasattr(self.llm, "ask"):
            answer = "\n".join([
                f"参考片段 [{i}]: {c.text[:200]}..." for i, c in enumerate(chunks, start=1)
            ])
            cites = [
                {
                    "marker": f"[{i}]",
                    "source": c.metadata.get("source"),
                    "doc_id": c.metadata.get("doc_id"),
                    "chunk_index": c.metadata.get("chunk_index"),
                }
                for i, c in enumerate(chunks, start=1)
            ]
            return QAResult(answer=answer, citations=cites)

        context = "".join([f"[{i}] {c.text}\n" for i, c in enumerate(chunks, start=1)])
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        # Import Message lazily to avoid hard runtime dependency during offline smoke
        from spoon_ai.chat import Message  # type: ignore
        messages = [
            Message(role="system", content=DEFAULT_QA_SYSTEM),
            Message(role="user", content=prompt),
        ]
        resp = await self.llm.ask(messages=messages)
        # Accept both ChatBot LLMResponse-like objects and plain strings
        if isinstance(resp, str):
            text = resp
        else:
            text = getattr(resp, "content", "") or ""

        cites: List[Dict] = []
        for i, c in enumerate(chunks, start=1):
            marker = f"[{i}]"
            if marker in text:
                cites.append(
                    {
                        "marker": marker,
                        "source": c.metadata.get("source"),
                        "doc_id": c.metadata.get("doc_id"),
                        "chunk_index": c.metadata.get("chunk_index"),
                    }
                )
        return QAResult(answer=text, citations=cites)
