from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import RagConfig
from .embeddings import EmbeddingClient
from .vectorstores import VectorStore

logger = logging.getLogger(__name__)


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
        self.bm25 = None
        self.bm25_data = None
        self._load_bm25()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text with CJK support.

        CJK ideographs are emitted as individual tokens;
        Latin/Cyrillic words are kept whole.
        """
        return re.findall(
            r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]|[a-zA-Z0-9\u00C0-\u024F]+',
            text.lower(),
        )

    def _load_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            bm25_file = os.path.join(self.config.rag_dir, "bm25_data.json")
            if os.path.exists(bm25_file):
                with open(bm25_file, "r", encoding="utf-8") as f:
                    self.bm25_data = json.load(f)

                tokenized_corpus = [self._tokenize(doc) for doc in self.bm25_data["texts"]]
                self.bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            pass  # BM25 optional
        except Exception as e:
            logger.warning("Failed to load BM25 index: %s", e)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievedChunk],
        bm25_results: List[RetrievedChunk],
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> List[RetrievedChunk]:
        """Calculates RRF score: weight * 1 / (k + rank).

        ``k`` (smoothing constant) is read from ``self.config.rrf_k``.
        """
        rrf_k = self.config.rrf_k
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results, 1):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0) + vector_weight * (1.0 / (rrf_k + rank))
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(bm25_results, 1):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0) + bm25_weight * (1.0 / (rrf_k + rank))
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

        final_results = [
            RetrievedChunk(id=cid, text=chunk_map[cid].text, score=score, metadata=chunk_map[cid].metadata)
            for cid, score in fused_scores.items()
        ]
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results

    def retrieve(
        self,
        query: str,
        *,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or self.config.top_k
        threshold = min_similarity if min_similarity is not None else self.config.min_similarity
        overfetch = max(k * self.config.retrieval_overfetch_factor, 20)

        # 1. Vector Search — collect ALL results; threshold applied later
        query_vec = self.embeddings.embed([query])
        if not query_vec:
            raise ValueError("Embedding failed: got empty result for query")
        query_results = self.store.query(
            collection=collection or self.config.collection,
            query_embeddings=query_vec,
            top_k=overfetch,
        )
        raw = query_results[0] if query_results else []

        vector_chunks: List[RetrievedChunk] = []
        for id_, score, md in raw:
            text = md.get("text", "")
            vector_chunks.append(RetrievedChunk(id=id_, text=text, score=score, metadata=md))

        # 2. BM25 Search (optional)
        bm25_chunks: List[RetrievedChunk] = []
        if self.bm25 and self.bm25_data:
            try:
                tokenized_query = self._tokenize(query)
                scores = self.bm25.get_scores(tokenized_query)
                top_n_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[:overfetch]

                for idx in top_n_indices:
                    if scores[idx] <= 0:
                        continue
                    bm25_chunks.append(RetrievedChunk(
                        id=self.bm25_data["ids"][idx],
                        text=self.bm25_data["texts"][idx],
                        score=scores[idx],
                        metadata=self.bm25_data["metadatas"][idx],
                    ))
            except Exception as e:
                logger.warning("BM25 search failed: %s", e)

        # 3. Fusion or pure-vector fallback
        if bm25_chunks:
            # RRF is rank-based — threshold filtering is not meaningful here,
            # so we fuse first and rely on rank ordering.
            chunks = self._reciprocal_rank_fusion(vector_chunks, bm25_chunks)
        else:
            # Pure-vector path: apply similarity threshold
            chunks = [c for c in vector_chunks if c.score >= threshold]

        # Lightweight dedup by text
        seen: set[str] = set()
        deduped: List[RetrievedChunk] = []
        for c in chunks:
            key = c.text.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        return deduped[:k]

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        for i, c in enumerate(chunks, start=1):
            src = c.metadata.get("source", "")
            lines.append(f"[{i}] {c.text}\n(source: {src})\n")
        return "\n".join(lines)
