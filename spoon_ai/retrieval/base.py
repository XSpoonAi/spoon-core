from typing import List, Dict, Any

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class BaseRetrievalClient:
    """
    Abstract base class for retrieval clients.
    """
    def add_documents(self, documents: List[Document]):
        raise NotImplementedError

    def query(self, query: str, k: int = 10) -> List[Document]:
        raise NotImplementedError

    def delete_collection(self):
        raise NotImplementedError
