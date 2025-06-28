# Retrieval package for SpoonAI 
from .base import BaseRetrievalClient, Document
from .chroma import ChromaClient
from .qdrant import QdrantClient

# Factory for retrieval client
RETRIEVAL_CLIENTS = {
    'chroma': ChromaClient,
    'qdrant': QdrantClient,
}

def get_retrieval_client(backend: str = 'chroma', **kwargs) -> BaseRetrievalClient:
    if backend not in RETRIEVAL_CLIENTS or RETRIEVAL_CLIENTS[backend] is None:
        raise ValueError(f"Retrieval backend '{backend}' is not available.")
    return RETRIEVAL_CLIENTS[backend](**kwargs)