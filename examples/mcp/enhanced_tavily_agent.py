"""
Enhanced Tavily Search Agent based on the new registration system
Demonstrates how to use EnhancedBaseAgent to create feature-rich agents
"""

import os
import logging
from typing import Dict, Any, Optional
from fastmcp.client.transports import StdioTransport

from spoon_ai.agents.enhanced_base import EnhancedBaseAgent
from spoon_ai.chat import ChatBot

logger = logging.getLogger(__name__)


class EnhancedTavilyAgent(EnhancedBaseAgent):
    """
    Enhanced Tavily Search Agent based on the new agent registration system
    Provides intelligent web search functionality with support for multiple search types and result processing
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set basic agent information
        self.name = "EnhancedTavilyAgent"
        self.description = (
            "Enhanced Tavily search agent providing intelligent web search, news search, "
            "research information collection and other functions with context-aware result processing"
        )
        
        # Tavily-specific configuration
        self.api_key = kwargs.get('api_key', os.getenv('TAVILY_API_KEY'))
        self.max_results = kwargs.get('max_results', 5)
        self.search_depth = kwargs.get('search_depth', 'basic')
        
        # Set system prompt
        self.system_prompt = """
        You are the Enhanced TavilySearchAgent, a professional web search AI assistant.
        
        Your core capabilities:
        1. Intelligent Web Search - Use Tavily API for high-quality web searches
        2. News and Current Events Search - Get the latest news and current information
        3. Research Information Collection - Deep collection of research materials on specific topics
        4. Context-Aware Processing - Optimize search strategies based on conversation context
        5. Intelligent Result Organization - Analyze and summarize search results
        
        Search Strategies:
        - For general queries, use basic search mode
        - For news queries, prioritize searching for latest information
        - For research queries, perform deep searches and provide multi-perspective information
        - For technical queries, seek authoritative sources and best practices
        
        Always provide accurate, timely, relevant information and cite information sources.
        If search results are not ideal, will try to optimize search keywords and search again.
        """
        
        self.next_step_prompt = """
        Based on the user's query and existing search results, decide the next action:
        1. If more information is needed, optimize search keywords and search again
        2. If results are sufficient, organize and summarize information
        3. If the query is unclear, request user clarification
        4. If specific types of search are needed (such as news, academic), adjust search strategy
        """
    
    async def _initialize_agent(self) -> None:
        """
        Initialize Tavily agent-specific logic
        """
        # Validate API key
        if not self.api_key or self.api_key == "your-api-key-here":
            logger.warning(
                "Tavily API key not set or using default value. "
                "Please set environment variable TAVILY_API_KEY or specify api_key in configuration."
            )
        else:
            logger.info("Tavily API key configured")
        
        # If MCP transport configuration exists, verify connection
        if hasattr(self, 'list_mcp_tools'):
            try:
                tools = await self.list_mcp_tools()
                tavily_tools = [tool for tool in tools if 'tavily' in tool.name.lower()]
                logger.info(f"Tavily agent initialization complete, available tools: {len(tavily_tools)}")
                
                if not tavily_tools:
                    logger.warning("No Tavily-related MCP tools found")
            except Exception as e:
                logger.warning(f"Error initializing Tavily MCP tools: {e}")
        
        # Initialize search history
        self.search_history = []
        self.context_keywords = set()
    
    async def step(self) -> str:
        """
        Execute one step of the Tavily agent
        """
        messages = self.memory.get_messages()
        if not messages:
            return "No messages to process"
        
        last_message = messages[-1]
        if last_message.role.value != "user":
            return "Waiting for user input"
        
        user_query = last_message.content
        
        try:
            # Analyze query type
            query_type = self._analyze_query_type(user_query)
            logger.info(f"Query type: {query_type}")
            
            # Extract keywords and update context
            keywords = self._extract_keywords(user_query)
            self.context_keywords.update(keywords)
            
            # Execute corresponding search strategy based on query type
            if query_type == "news":
                result = await self._search_news(user_query)
            elif query_type == "research":
                result = await self._research_search(user_query)
            elif query_type == "technical":
                result = await self._technical_search(user_query)
            else:
                result = await self._general_search(user_query)
            
            # Process and format results
            formatted_response = await self._format_search_response(result, user_query, query_type)
            
            # Add to memory
            self.add_message("assistant", formatted_response)
            
            # Update search history
            self.search_history.append({
                "query": user_query,
                "type": query_type,
                "keywords": keywords,
                "result_summary": result[:200] if isinstance(result, str) else "Search completed"
            })
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"Error during search process: {str(e)}"
            logger.error(error_msg)
            self.add_message("assistant", error_msg)
            return error_msg
    
    def _analyze_query_type(self, query: str) -> str:
        """
        Analyze query type to determine search strategy
        """
        query_lower = query.lower()
        
        # News-related keywords
        news_keywords = ['news', 'latest', 'recent', 'today', 'yesterday', 'this week', 'breaking', 'current']
        if any(keyword in query_lower for keyword in news_keywords):
            return "news"
        
        # Research-related keywords
        research_keywords = ['research', 'study', 'analysis', 'report', 'paper', 'survey', 'statistics', 'academic']
        if any(keyword in query_lower for keyword in research_keywords):
            return "research"
        
        # Technical-related keywords
        tech_keywords = ['tutorial', 'how to', 'configure', 'install', 'code', 'programming', 'development', 'api', 'documentation']
        if any(keyword in query_lower for keyword in tech_keywords):
            return "technical"
        
        return "general"
    
    def _extract_keywords(self, query: str) -> set:
        """
        Extract keywords from query
        """
        # Simple keyword extraction (more complex NLP methods can be used in practice)
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        # Filter common stop words
        stop_words = {'the', 'is', 'in', 'and', 'or', 'but', 'if', 'because', 'a', 'an', 'to', 'for', 'of', 'with', 'by'}
        keywords = {word for word in words if len(word) > 2 and word not in stop_words}
        return keywords
    
    async def _search_news(self, query: str) -> str:
        """
        Execute news search
        """
        if hasattr(self, 'call_mcp_tool'):
            try:
                # Use news-specific search parameters
                result = await self.call_mcp_tool(
                    'tavily_search',
                    query=query,
                    search_depth='basic',
                    include_domains=['news.google.com', 'reuters.com', 'bbc.com', 'cnn.com'],
                    max_results=self.max_results
                )
                return result
            except Exception as e:
                logger.error(f"News search failed: {e}")
                return f"News search failed: {str(e)}"
        else:
            return "MCP tools not available, cannot execute news search"
    
    async def _research_search(self, query: str) -> str:
        """
        Execute research deep search
        """
        if hasattr(self, 'call_mcp_tool'):
            try:
                # Use deep search parameters
                result = await self.call_mcp_tool(
                    'tavily_search',
                    query=query,
                    search_depth='advanced',
                    include_domains=['scholar.google.com', 'arxiv.org', 'researchgate.net'],
                    max_results=self.max_results * 2  # Research search needs more results
                )
                return result
            except Exception as e:
                logger.error(f"Research search failed: {e}")
                return f"Research search failed: {str(e)}"
        else:
            return "MCP tools not available, cannot execute research search"
    
    async def _technical_search(self, query: str) -> str:
        """
        Execute technical search
        """
        if hasattr(self, 'call_mcp_tool'):
            try:
                # Use technical-specific search parameters
                result = await self.call_mcp_tool(
                    'tavily_search',
                    query=query,
                    search_depth='basic',
                    include_domains=['stackoverflow.com', 'github.com', 'docs.python.org', 'developer.mozilla.org'],
                    max_results=self.max_results
                )
                return result
            except Exception as e:
                logger.error(f"Technical search failed: {e}")
                return f"Technical search failed: {str(e)}"
        else:
            return "MCP tools not available, cannot execute technical search"
    
    async def _general_search(self, query: str) -> str:
        """
        Execute general search
        """
        if hasattr(self, 'call_mcp_tool'):
            try:
                result = await self.call_mcp_tool(
                    'tavily_search',
                    query=query,
                    search_depth=self.search_depth,
                    max_results=self.max_results
                )
                return result
            except Exception as e:
                logger.error(f"General search failed: {e}")
                return f"Search failed: {str(e)}"
        else:
            return "MCP tools not available, cannot execute search"
    
    async def _format_search_response(self, search_result: str, original_query: str, query_type: str) -> str:
        """
        Format search response using LLM for intelligent organization
        """
        try:
            formatting_prompt = f"""
            Please provide a structured answer for the user query based on the following search results.
            
            User Query: {original_query}
            Query Type: {query_type}
            Search Results: {search_result}
            
            Please organize the answer in the following format:
            1. Brief Summary (2-3 sentences)
            2. Detailed Information (list main content in bullet points)
            3. Information Sources (if available)
            4. Related Suggestions or Follow-up Actions (if applicable)
            
            Keep the answer accurate, useful and easy to understand.
            """
            
            formatted_response = await self.llm.achat(
                messages=[{"role": "user", "content": formatting_prompt}],
                system_prompt="You are an information organization expert, skilled at converting search results into clear, useful answers."
            )
            
            return formatted_response
            
        except Exception as e:
            logger.warning(f"Failed to format answer, returning raw results: {e}")
            return f"Search Results:\n{search_result}"
    
    async def get_search_history(self) -> list:
        """
        Get search history
        """
        return self.search_history
    
    async def get_context_keywords(self) -> set:
        """
        Get context keywords
        """
        return self.context_keywords
    
    async def clear_context(self) -> None:
        """
        Clear search context
        """
        self.search_history.clear()
        self.context_keywords.clear()
        logger.info("Search context cleared")


# Simplified configuration function for registration system
def create_enhanced_tavily_agent(**kwargs) -> EnhancedTavilyAgent:
    """
    Factory function to create Enhanced Tavily Agent
    """
    # Auto-configure MCP transport
    if 'mcp_transport' not in kwargs:
        stdio_transport = StdioTransport(
            command="npx",
            args=["-y", "tavily-mcp"],
            env={
                "TAVILY_API_KEY": kwargs.get('api_key', os.getenv("TAVILY_API_KEY", "your-api-key-here"))
            }
        )
        kwargs['mcp_transport'] = stdio_transport
    
    # Set default LLM (if not provided)
    if 'llm' not in kwargs:
        kwargs['llm'] = ChatBot(
            llm_provider="openai",
            model_name="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1"
        )
    
    return EnhancedTavilyAgent(**kwargs) 