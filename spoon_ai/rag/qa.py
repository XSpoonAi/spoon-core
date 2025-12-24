from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING, Any, Set

if TYPE_CHECKING:
    from spoon_ai.chat import Message  # type: ignore

from .config import RagConfig
from .retriever import RetrievedChunk


@dataclass
class Citation:
    source: str
    marker: str
    doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    text_snippet: Optional[str] = None


@dataclass
class QAResult:
    answer: str
    citations: List[Citation]
    raw_response: Optional[Any] = None


DEFAULT_QA_SYSTEM = (
    "You are a helpful assistant that answers questions using the provided context. "
    "Always cite sources using the exact [id] markers provided in the context (e.g. [docname_0], [url_1])."
)

QA_PROMPT_TEMPLATE = (
    "Answer the user question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Use the provided [id] markers in the answer to cite the snippets exactly.\n"
    "- Keep the answer concise and relevant.\n"
)


class RagQA:
    def __init__(
        self,
        *,
        config: RagConfig,
        llm: Any,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        self.config = config
        self.llm = llm
        self.system_prompt = system_prompt or DEFAULT_QA_SYSTEM
        self.user_template = user_template or QA_PROMPT_TEMPLATE
        # Simple char limit safeguard (approx 30k tokens for modern models, but keep it safe)
        self.max_context_chars = 60000

    def _get_chunk_marker(self, chunk: RetrievedChunk) -> str:
        """Generate a stable citation marker: [doc_id_chunk_index]"""
        raw_id = str(chunk.metadata.get("doc_id", "unknown"))
        # Clean doc_id to be shorter and safer
        # 1. Get basename if it looks like a path
        if "/" in raw_id or "\\" in raw_id:
            try:
                raw_id = os.path.basename(str(raw_id))
            except Exception:
                pass
        
        # 2. Remove extension for brevity
        base = os.path.splitext(raw_id)[0]
        
        # 3. Sanitize characters
        clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", base)
        
        idx = chunk.metadata.get("chunk_index", "0")
        return f"[{clean_id}_{idx}]"

    def _truncate_context(self, chunks: List[RetrievedChunk]) -> str:
        """Join chunks into a context string using stable IDs."""
        lines = []
        current_len = 0
        
        for c in chunks:
            marker = self._get_chunk_marker(c)
            # Format: [doc_1] content...
            snippet = f"{marker} {c.text}"
            snippet_len = len(snippet) + 2  # + 2 for newlines
            
            if current_len + snippet_len > self.max_context_chars:
                # Stop adding chunks if we exceed the budget
                break
                
            lines.append(snippet)
            current_len += snippet_len

        return "\n\n".join(lines)

    async def answer(self, question: str, chunks: List[RetrievedChunk]) -> QAResult:
        # P1: Handle empty chunks
        if not chunks:
            return QAResult(
                answer="I cannot answer this question because no relevant documents were found.",
                citations=[]
            )

        # Build map for citation lookup
        chunk_map = {self._get_chunk_marker(c): c for c in chunks}

        # Optional offline fallback
        if os.getenv("RAG_FAKE_QA") == "1" or not (self.llm and hasattr(self.llm, "ask")):
            # P2: Consistent language (English default) for offline fallback to match system prompt
            answer = "Offline Mode / No LLM:\n" + "\n".join([
                f"Source {self._get_chunk_marker(c)}: {c.text[:200]}..." for c in chunks
            ])
            cites = [
                Citation(
                    marker=self._get_chunk_marker(c),
                    source=c.metadata.get("source", "unknown"),
                    doc_id=c.metadata.get("doc_id"),
                    chunk_index=c.metadata.get("chunk_index"),
                    text_snippet=c.text[:50]
                )
                for c in chunks
            ]
            return QAResult(answer=answer, citations=cites)

        # P0 & P1: Truncate and clean join
        context = self._truncate_context(chunks)
        prompt = self.user_template.format(context=context, question=question)

        # Lazy import to avoid circular dependency
        from spoon_ai.chat import Message  # type: ignore
        
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        
        resp = await self.llm.ask(messages=messages)
        
        if isinstance(resp, str):
            text = resp
        else:
            text = getattr(resp, "content", "") or ""

        # P1: ID-based citation parsing
        # Matches [doc_1], [file_name_12], etc.
        final_citations: List[Citation] = []
        seen_markers: Set[str] = set()
        
        # Regex to find potential markers in the text
        # We look for [content] and check if it exists in our map
        matches = re.findall(r"\[([^\]]+)\]", text)
        
        for m_str in matches:
            marker = f"[{m_str}]"
            if marker in chunk_map and marker not in seen_markers:
                c = chunk_map[marker]
                seen_markers.add(marker)
                final_citations.append(
                    Citation(
                        marker=marker,
                        source=c.metadata.get("source", "unknown"),
                        doc_id=c.metadata.get("doc_id"),
                        chunk_index=c.metadata.get("chunk_index"),
                        text_snippet=c.text[:100]  # Store a bit of text for verification
                    )
                )

        return QAResult(answer=text, citations=final_citations, raw_response=resp)
