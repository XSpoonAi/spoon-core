import os
from typing import List, Dict, Any
import uuid
import openai
from .base import BaseRetrievalClient, Document

class ChromaClient(BaseRetrievalClient):
    def __init__(self, config_dir: str):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Chroma is not installed. Please install it with 'pip install chromadb'.")
        
        self.client = chromadb.PersistentClient(path=os.path.join(config_dir, "spoon_ai.db"))
        self.collection = self.client.get_or_create_collection("spoon_ai")
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI()
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's API directly"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the collection"""
        for doc in documents:
            # TODO: parallelize this
            doc_embedding = self._get_embedding(doc.page_content)
            self.collection.add(
                ids=[doc.metadata.get("id", str(uuid.uuid4()))],
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                embeddings=[doc_embedding]
            )
        
    def query(self, query: str, k: int = 10) -> List[Document]:
        """Query the collection"""
        query_embedding = self._get_embedding(query)
        results = self.collection.query(query_embedding, n_results=k)
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append(Document(page_content=results["documents"][0][i], metadata=results["metadatas"][0][i]))
        return docs

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection.name)
