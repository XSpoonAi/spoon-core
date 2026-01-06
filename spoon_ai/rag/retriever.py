from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import RagConfig
from .embeddings import EmbeddingClient
from .config import RagConfig
from .embeddings import EmbeddingClient
from .vectorstores import VectorStore
import os
import pickle


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
        query_vec = self.embeddings.embed([query])
        raw = self.store.query(
            collection=collection or self.config.collection,
            query_embeddings=query_vec,
            top_k=max(k * 2, k),  # small overfetch for lightweight dedup/MMR
        )[0]
        # Build chunks
        chunks: List[RetrievedChunk] = []
        for id_, score, md in raw:
            if score < threshold:
                continue
            text = md.get("text", "")
            chunks.append(RetrievedChunk(id=id_, text=text, score=score, metadata=md))

        # Hybrid Search: Add BM25 results
        if self.bm25 and self.bm25_data:
            try:
                tokenized_query = query.lower().split()
                # Get indices of top k results
                # We fetch top_k indices. rank_bm25 returns the actual documents by get_top_n, 
                # but we need indices to look up metadata.
                # So we calculate scores and sort manually or use private API.
                # Standard way: get scores
                scores = self.bm25.get_scores(tokenized_query)
                # Get top k indices
                top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
                
                for idx in top_n_indices:
                    # Skip if score is 0 (no match)
                    if scores[idx] <= 0:
                        continue
                        
                    c_id = self.bm25_data["ids"][idx]
                    # If not already present
                    if not any(c.id == c_id for c in chunks):
                        c_text = self.bm25_data["texts"][idx]
                        c_meta = self.bm25_data["metadatas"][idx]
                        # Boost score for keyword match to prioritize it
                        # Or assign a high constant like 0.95
                        chunks.append(RetrievedChunk(
                            id=c_id,
                            text=c_text,
                            score=0.95, 
                            metadata=c_meta
                        ))
            except Exception as e:
                print(f"[Warning] BM25 search failed: {e}") 

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

