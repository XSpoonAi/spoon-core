from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import RagConfig
from .embeddings import EmbeddingClient
from .vectorstores import VectorStore


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: Dict


class RagRetriever:
    def __init__(
        self,
        *,
        config: RagConfig,
        store: VectorStore,
        embeddings: EmbeddingClient,
    ) -> None:
        self.config = config
        self.store = store
        self.embeddings = embeddings

    def retrieve(
        self,
        query: str,
        *,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or self.config.top_k
        query_vec = self.embeddings.embed([query])
        raw = self.store.query(
            collection=collection or self.config.collection,
            query_embeddings=query_vec,
            top_k=max(k * 2, k),  # small overfetch for lightweight dedup/MMR
        )[0]
        # Build chunks
        chunks: List[RetrievedChunk] = []
        for id_, score, md in raw:
            text = md.get("text", "")
            chunks.append(RetrievedChunk(id=id_, text=text, score=score, metadata=md))

        # Lightweight dedup by text
        seen = set()
        deduped: List[RetrievedChunk] = []
        for c in chunks:
            key = c.text.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        # Naive diversity: sort by score, but limit identical sources in immediate sequence
        deduped.sort(key=lambda x: x.score, reverse=True)
        return deduped[:k]

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        for i, c in enumerate(chunks, start=1):
            src = c.metadata.get("source", "")
            lines.append(f"[{i}] {c.text}\n(source: {src})\n")
        return "\n".join(lines)

