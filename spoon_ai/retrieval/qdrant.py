import os
from typing import List, Dict, Any, Optional
import uuid
import openai
from .base import BaseRetrievalClient, Document


class QdrantClient(BaseRetrievalClient):
    def __init__(
        self,
        collection_name: str = "spoon_ai",
        config_dir: Optional[str] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
    ):
        try:
            from qdrant_client import QdrantClient as Qdrant
        except ImportError:
            raise ImportError(
                "Qdrant client is not installed. Please install it with 'pip install qdrant-client'."
            )

        self.qdrant = Qdrant(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=config_dir,
        )
        self.collection_name = collection_name
        self.openai_client = openai.OpenAI()
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.http import models

        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
            )

    def _get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    def add_documents(self, documents: List[Document]):
        from qdrant_client import models

        points = []
        for doc in documents:
            doc_embedding = self._get_embedding(doc.page_content)
            point_id = doc.metadata.get("id", str(uuid.uuid4()))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=doc_embedding,
                    payload={"text": doc.page_content, **doc.metadata},
                )
            )
        self.qdrant.upsert(collection_name=self.collection_name, points=points)

    def query(self, query: str, k: int = 10) -> List[Document]:
        query_embedding = self._get_embedding(query)
        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            with_payload=True,
        ).points
        docs = []
        for hit in search_result:
            payload = hit.payload or {}
            docs.append(
                Document(page_content=payload.pop("text", ""), metadata=payload)
            )
        return docs

    def delete_collection(self):
        self.qdrant.delete_collection(self.collection_name)
