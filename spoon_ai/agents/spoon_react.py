from typing import List, Union, Any, Dict, Optional
import asyncio
from fastmcp.client.transports import (FastMCPTransport, PythonStdioTransport,
                                       SSETransport, WSTransport, NpxStdioTransport,
                                       FastMCPStdioTransport, UvxStdioTransport)
from fastmcp.client import Client as MCPClient
from pydantic import Field
import logging

from spoon_ai.chat import ChatBot
from spoon_ai.prompts.spoon_react import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from spoon_ai.tools import ToolManager


from .toolcall import ToolCallAgent

logger = logging.getLogger(__name__)

def create_configured_chatbot():
    """Create a ChatBot instance with intelligent provider selection."""
    from spoon_ai.llm.config import ConfigurationManager

    # Get the optimal provider based on configuration and availability
    try:
        config_manager = ConfigurationManager()
        optimal_provider = config_manager.get_default_provider()

        logger.info(f"Creating ChatBot with optimal provider: {optimal_provider}")

        # Use the LLM manager architecture with the selected provider
        return ChatBot(llm_provider=optimal_provider)

    except Exception as e:
        logger.error(f"Failed to initialize ChatBot with LLM manager: {e}")
        raise RuntimeError(f"Failed to initialize ChatBot: {e}") from e

class SpoonReactAI(ToolCallAgent):

    name: str = "spoon_react"
    description: str = "A smart ai agent in neo blockchain"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 10
    tool_choice: str = "auto"

    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    llm: ChatBot = Field(default_factory=create_configured_chatbot)

    mcp_transport: Union[str, WSTransport, SSETransport, PythonStdioTransport, NpxStdioTransport, FastMCPTransport, FastMCPStdioTransport, UvxStdioTransport] = Field(default="mcp_server")
    mcp_topics: List[str] = Field(default=["spoon_react"])

    def __init__(self, **kwargs):
        """Initialize SpoonReactAI with both ToolCallAgent and MCPClientMixin initialization"""
        # Call parent class initializers
        ToolCallAgent.__init__(self, **kwargs)

    async def initialize(self, __context: Any = None):
        """Initialize async components and subscribe to topics"""
        logger.info(f"Initializing SpoonReactAI agent '{self.name}'")

        # First establish connection to MCP server
        try:
            # Verify connection
            await self.connect()

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {str(e)}")
            # If context has error handling, use it
            if __context and hasattr(__context, 'report_error'):
                await __context.report_error(e)
            raise

    
    @property
    def avaliable_tools(self):
        """Backward compatibility property for misspelled field name."""
        import warnings
        warnings.warn(
            "avaliable_tools is deprecated and will be removed in v2.0. "
            "Use available_tools instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.available_tools

    @avaliable_tools.setter
    def avaliable_tools(self, value):
        """Backward compatibility setter for misspelled field name."""
        import warnings
        warnings.warn(
            "avaliable_tools is deprecated and will be removed in v2.0. "
            "Use available_tools instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.available_tools = value