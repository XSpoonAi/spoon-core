import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from logging import getLogger

logger = getLogger(__name__)

class RetrievalMixin:
    """
    Enhanced mixin class for retrieval-augmented generation functionality.
    
    Provides async-compatible RAG operations with robust error handling,
    proper configuration management, and document validation.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize retrieval-related attributes"""
        super().__init__(*args, **kwargs)
        self._retrieval_client = None
        self._retrieval_backend = None
        self._retrieval_config = {}
        
    @property
    def config_dir(self) -> Path:
        """Get configuration directory with fallback options"""
        # Try multiple sources for config directory
        if hasattr(self, '_config_dir') and self._config_dir:
            return Path(self._config_dir)
        
        if hasattr(self, 'name'):
            agent_config_dir = Path.cwd() / "config" / f"agent_{self.name}"
            agent_config_dir.mkdir(parents=True, exist_ok=True)
            return agent_config_dir
            
        # Fallback to default
        default_config = Path.cwd() / "config" / "retrieval"
        default_config.mkdir(parents=True, exist_ok=True)
        return default_config

    async def initialize_retrieval_client(self, backend: str = 'chroma', **kwargs) -> bool:
        """
        Initialize the retrieval client asynchronously
        
        Args:
            backend: Retrieval backend to use ('chroma', 'faiss', etc.)
            **kwargs: Additional configuration parameters
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Avoid re-initialization if already using same backend
            if (self._retrieval_client is not None and 
                self._retrieval_backend == backend and
                self._retrieval_config == kwargs):
                logger.debug(f"Retrieval client already initialized with backend: {backend}")
                return True
                
            logger.info(f"Initializing retrieval client with backend: {backend}")
            
            # Import here to avoid circular imports
            from spoon_ai.retrieval import get_retrieval_client
            
            # Prepare configuration
            config = {
                'config_dir': str(self.config_dir),
                **kwargs
            }
            
            # Initialize client (run in thread pool if synchronous)
            if asyncio.iscoroutinefunction(get_retrieval_client):
                self._retrieval_client = await get_retrieval_client(backend, **config)
            else:
                # Run synchronous operation in thread pool
                self._retrieval_client = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: get_retrieval_client(backend, **config)
                )
            
            # Cache configuration
            self._retrieval_backend = backend
            self._retrieval_config = kwargs.copy()
            
            agent_name = getattr(self, 'name', 'unknown')
            logger.info(f"✅ Retrieval client initialized for agent '{agent_name}' with backend: {backend}")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Failed to import retrieval client: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval client: {e}")
            return False

    def _validate_documents(self, documents: List[Any]) -> bool:
        """Validate document format and content"""
        if not isinstance(documents, list):
            logger.error("Documents must be provided as a list")
            return False
            
        if not documents:
            logger.warning("Empty document list provided")
            return True  # Empty list is valid, just warn
            
        # Check first few documents for common attributes
        for i, doc in enumerate(documents[:3]):  # Check first 3 docs
            if not hasattr(doc, 'page_content') and not hasattr(doc, 'content'):
                logger.error(f"Document {i} missing required 'page_content' or 'content' attribute")
                return False
                
        logger.debug(f"✅ Validated {len(documents)} documents")
        return True

    async def add_documents(self, documents: List[Any], backend: str = 'chroma', **kwargs) -> bool:
        """
        Add documents to the retrieval system asynchronously
        
        Args:
            documents: List of documents to add
            backend: Retrieval backend to use
            **kwargs: Additional parameters
            
        Returns:
            bool: True if documents added successfully, False otherwise
        """
        if not self._validate_documents(documents):
            return False
            
        # Initialize client if needed
        if not await self.initialize_retrieval_client(backend, **kwargs):
            logger.error("❌ Failed to initialize retrieval client for adding documents")
            return False
            
        try:
            # Add documents (run in thread pool if synchronous)
            if asyncio.iscoroutinefunction(self._retrieval_client.add_documents):
                await self._retrieval_client.add_documents(documents)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._retrieval_client.add_documents, documents
                )
                
            agent_name = getattr(self, 'name', 'unknown')
            logger.info(f"✅ Added {len(documents)} documents to retrieval system for agent '{agent_name}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False

    async def retrieve_relevant_documents(self, query: str, k: int = 5, backend: str = 'chroma', **kwargs) -> List[Any]:
        """
        Retrieve relevant documents for a query asynchronously
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            backend: Retrieval backend to use
            **kwargs: Additional parameters
            
        Returns:
            List of relevant documents (empty list on error)
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for document retrieval")
            return []
            
        # Initialize client if needed
        if not await self.initialize_retrieval_client(backend, **kwargs):
            logger.error("❌ Failed to initialize retrieval client for query")
            return []
            
        try:
            # Query documents (run in thread pool if synchronous)
            if asyncio.iscoroutinefunction(self._retrieval_client.query):
                docs = await self._retrieval_client.query(query, k=k)
            else:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._retrieval_client.query(query, k=k)
                )
                
            agent_name = getattr(self, 'name', 'unknown')
            logger.debug(f"🔍 Retrieved {len(docs)} documents for query in agent '{agent_name}': {query[:50]}...")
            return docs if docs else []
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            return []

    async def get_context_from_query(self, query: str, k: int = 5, max_context_length: int = 4000, **kwargs) -> tuple[str, List[Any]]:
        """
        Get context string from relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            max_context_length: Maximum length of context string
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Tuple of (context_string, relevant_documents)
        """
        relevant_docs = await self.retrieve_relevant_documents(query, k=k, **kwargs)
        
        if not relevant_docs:
            logger.debug(f"No relevant documents found for query: {query[:50]}...")
            return "", []
        
        # Build context string with length limits
        context_str = "\n\nRelevant context:\n"
        total_length = len(context_str)
        included_docs = []
        
        for i, doc in enumerate(relevant_docs):
            # Get document content
            doc_content = getattr(doc, 'page_content', getattr(doc, 'content', str(doc)))
            
            # Format document section
            doc_section = f"[Document {i+1}]\n{doc_content}\n\n"
            
            # Check if adding this document would exceed limit
            if total_length + len(doc_section) > max_context_length:
                logger.debug(f"Context length limit reached, included {len(included_docs)}/{len(relevant_docs)} documents")
                break
                
            context_str += doc_section
            total_length += len(doc_section)
            included_docs.append(doc)
        
        agent_name = getattr(self, 'name', 'unknown')        
        logger.info(f"📄 Generated context ({total_length} chars) from {len(included_docs)} documents for agent '{agent_name}'")
        
        return context_str, included_docs

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        stats = {
            'client_initialized': self._retrieval_client is not None,
            'backend': self._retrieval_backend,
            'config_dir': str(self.config_dir),
        }
        
        # Try to get additional stats from client
        if self._retrieval_client and hasattr(self._retrieval_client, 'get_stats'):
            try:
                stats.update(self._retrieval_client.get_stats())
            except Exception as e:
                logger.debug(f"Could not get retrieval client stats: {e}")
                
        return stats

    def clear_retrieval_cache(self):
        """Clear retrieval client and reset state"""
        if self._retrieval_client and hasattr(self._retrieval_client, 'close'):
            try:
                self._retrieval_client.close()
            except Exception as e:
                logger.debug(f"Error closing retrieval client: {e}")
                
        self._retrieval_client = None
        self._retrieval_backend = None
        self._retrieval_config = {}
        
        agent_name = getattr(self, 'name', 'unknown')
        logger.debug(f"🧹 Cleared retrieval cache for agent '{agent_name}'")