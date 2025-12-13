from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import RagConfig
from .embeddings import EmbeddingClient
from .loader import load_inputs, chunk_text
from .vectorstores import VectorStore


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
        docs = load_inputs(inputs)
        records: List[IndexedRecord] = []
        for d in docs:
            chunks = chunk_text(d.text, self.config.chunk_size, self.config.chunk_overlap)
            for i, ch in enumerate(chunks):
                rec_id = str(uuid.uuid4())
                md = {
                    "source": d.source,
                    "doc_id": d.id,
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
        return len(records)

    def clear(self, *, collection: Optional[str] = None) -> None:
        self.store.delete_collection(collection or self.config.collection)

