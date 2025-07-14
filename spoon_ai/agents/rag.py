from typing import List, Optional, Dict, Any

from logging import getLogger
from spoon_ai.retrieval import get_retrieval_client

logger = getLogger(__name__)

DEBUG = False

def debug_log(message):
    if DEBUG:
        logger.debug(message)

class RetrievalMixin:
    """Mixin class for retrieval-augmented generation functionality"""
    
    def initialize_retrieval_client(self, backend: str = 'chroma', **kwargs):
        """Initialize the retrieval client if it doesn't exist"""
        if not hasattr(self, 'retrieval_client') or self.retrieval_client is None:
            debug_log(f"Initializing retrieval client with backend: {backend}")
            self.retrieval_client = get_retrieval_client(backend, config_dir=str(self.config_dir), **kwargs)
    
    def add_documents(self, documents, backend: str = 'chroma', **kwargs):
        """Add documents to the retrieval system"""
        self.initialize_retrieval_client(backend, **kwargs)
        self.retrieval_client.add_documents(documents)
        debug_log(f"Added {len(documents)} documents to retrieval system for agent {self.name}")

    def retrieve_relevant_documents(self, query, k=5, backend: str = 'chroma', **kwargs):
        """Retrieve relevant documents for a query"""
        self.initialize_retrieval_client(backend, **kwargs)
        try:
            docs = self.retrieval_client.query(query, k=k)
            debug_log(f"Retrieved {len(docs)} documents for query: {query}...")
            return docs
        except Exception as e:
            debug_log(f"Error retrieving documents: {e}")
            return []
    
    def get_context_from_query(self, query):
        """Get context string from relevant documents for a query"""
        relevant_docs = self.retrieve_relevant_documents(query)
        context_str = ""
        debug_log(f"Retrieved {len(relevant_docs)} relevant documents")
        
        if relevant_docs:
            context_str = "\n\nRelevant context:\n"
            for i, doc in enumerate(relevant_docs):
                context_str += f"[Document {i+1}]\n{doc.page_content}\n\n"
                
        return context_str, relevant_docs