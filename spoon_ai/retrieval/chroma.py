import os
import uuid
from typing import List, Dict, Any
import openai
from .base import BaseRetrievalClient, Document

class ChromaClient(BaseRetrievalClient):
    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(self, config_dir: str):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("Chroma is not installed. Please install it with 'pip install chromadb'.") from e
        db_path = os.path.join(config_dir, "spoon_ai.db")
        self._chromadb = chromadb
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="spoon_ai")
        self.openai_client = openai.OpenAI()

    def _get_embedding(self, text: str) -> List[float]:
        resp = self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    def add_documents(self, documents: List[Document], batch_size: int = 32):
        if not documents:
            return
        for start in range(0, len(documents), batch_size):
            chunk = documents[start:start + batch_size]
            texts = [doc.page_content for doc in chunk]
            metadatas = [dict(doc.metadata or {}) for doc in chunk]
            ids = [(md.get("id") if isinstance(md.get("id"), str) else None) or str(uuid.uuid4()) for md in metadatas]
            embeddings = self._get_embeddings_batch(texts)
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def query(self, query: str, k: int = 10) -> List[Document]:
        query_embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["ids", "documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        docs = results.get("documents", [[]])[0] if results.get("documents") else []
        metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        out = []
        for i in range(min(len(docs), len(metas), len(ids))):
            meta = metas[i] or {}
            if "id" not in meta and i < len(ids):
                meta = dict(meta)
                meta["id"] = ids[i]
            out.append(Document(page_content=docs[i], metadata=meta))
        return out

    def delete_collection(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name)
