from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import RagConfig
from .embeddings import EmbeddingClient
from .parser import UnstructuredParser
from .chunk import chunk_text
from .vectorstores import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexedRecord:
    id: str
    text: str
    metadata: Dict


class RagIndex:
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

    def ingest(self, inputs: Iterable[str], *, collection: Optional[str] = None) -> int:
        # Use UnstructuredParser for document parsing
        parser = UnstructuredParser()
        docs = parser.parse(inputs)
        records: List[IndexedRecord] = []
        for doc in docs:
            logger.info("Indexing document: %s", doc.filepath)

            # Use recursive chunking directly on elements
            chunks = chunk_text(
                text='',  # Not used when elements provided
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                chunk_method='recursive',
                elements=doc.elements
            )

            for i, ch in enumerate(chunks):
                rec_id = str(uuid.uuid4())
                md = {
                    "source": doc.filepath,
                    "doc_id": doc.filename,
                    "chunk_index": i,
                }
                records.append(IndexedRecord(id=rec_id, text=ch, metadata=md))

        if not records:
            return 0

        embeddings = self.embeddings.embed([r.text for r in records])
        self.store.add(
            collection=collection or self.config.collection,
            ids=[r.id for r in records],
            embeddings=embeddings,
            metadatas=[r.metadata | {"text": r.text} for r in records],
        )

        # Save data for BM25 (Hybrid Search) using JSON (safe serialization)
        try:
            bm25_file = os.path.join(self.config.rag_dir, "bm25_data.json")
            if not os.path.exists(self.config.rag_dir):
                os.makedirs(self.config.rag_dir, exist_ok=True)

            existing_data = {"ids": [], "texts": [], "metadatas": []}
            if os.path.exists(bm25_file):
                try:
                    with open(bm25_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Corrupted BM25 data file, starting fresh")
                    existing_data = {"ids": [], "texts": [], "metadatas": []}

            existing_data["ids"].extend([r.id for r in records])
            existing_data["texts"].extend([r.text for r in records])
            existing_data["metadatas"].extend([r.metadata for r in records])

            with open(bm25_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save BM25 data: %s", e)

        return len(records)

    def clear(self, *, collection: Optional[str] = None) -> None:
        # Also clear BM25 data
        try:
            bm25_file = os.path.join(self.config.rag_dir, "bm25_data.json")
            if os.path.exists(bm25_file):
                os.remove(bm25_file)
        except Exception:
            pass
        self.store.delete_collection(collection or self.config.collection)
