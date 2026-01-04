from __future__ import annotations

import os
import pickle
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
        self.bm25 = None
        self.bm25_data = None
        self._load_bm25()

    def _load_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            bm2_file = os.path.join(self.config.rag_dir, "bm25_dump.pkl")
            if os.path.exists(bm2_file):
                with open(bm2_file, "rb") as f:
                    self.bm25_data = pickle.load(f)
                
                # Simple whitespace tokenization
                tokenized_corpus = [doc.lower().split() for doc in self.bm25_data["texts"]]
                self.bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            pass  # BM25 optional
        except Exception as e:
            print(f"[Warning] Failed to load BM25 index: {e}")

    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[RetrievedChunk], 
        bm25_results: List[RetrievedChunk], 
        k: int = 60,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[RetrievedChunk]:
        """
        Calculates RRF score: 1 / (k + rank)
        """
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # Process Vector Results
        for rank, chunk in enumerate(vector_results, 1):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0) + vector_weight * (1.0 / (k + rank))
            chunk_map[chunk.id] = chunk

        # Process BM25 Results
        for rank, chunk in enumerate(bm25_results, 1):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0) + bm25_weight * (1.0 / (k + rank))
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

        # Build final list
        final_results = []
        for chunk_id, score in fused_scores.items():
            c = chunk_map[chunk_id]
            # Update score to the fused score
            final_results.append(RetrievedChunk(
                id=c.id,
                text=c.text,
                score=score,
                metadata=c.metadata
            ))

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
        
        # 1. Vector Search
        query_vec = self.embeddings.embed([query])
        raw = self.store.query(
            collection=collection or self.config.collection,
            query_embeddings=query_vec,
            top_k=max(k * 3, 20),
        )[0]
        
        vector_chunks: List[RetrievedChunk] = []
        for id_, score, md in raw:
            if score < threshold:
                continue
            text = md.get("text", "")
            vector_chunks.append(RetrievedChunk(id=id_, text=text, score=score, metadata=md))

        # 2. BM25 Search
        bm25_chunks: List[RetrievedChunk] = []
        if self.bm25 and self.bm25_data:
            try:
                tokenized_query = query.lower().split()
                scores = self.bm25.get_scores(tokenized_query)
                top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(k * 3, 20)]
                
                for idx in top_n_indices:
                    if scores[idx] <= 0:
                        continue
                    bm25_chunks.append(RetrievedChunk(
                        id=self.bm25_data["ids"][idx],
                        text=self.bm25_data["texts"][idx],
                        score=scores[idx],
                        metadata=self.bm25_data["metadatas"][idx]
                    ))
            except Exception as e:
                print(f"[Warning] BM25 search failed: {e}")

        # 3. Fusion
        if bm25_chunks:
            chunks = self._reciprocal_rank_fusion(vector_chunks, bm25_chunks)
        else:
            chunks = vector_chunks

        # Lightweight dedup by text
        seen = set()
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


