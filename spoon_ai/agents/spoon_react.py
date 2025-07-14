from typing import List, Union, Any, Dict, Optional
import asyncio
from fastmcp.client.transports import (FastMCPTransport, PythonStdioTransport,
                                       SSETransport, WSTransport)
from fastmcp.client import Client as MCPClient
from pydantic import Field
import logging

from spoon_ai.chat import ChatBot
from spoon_ai.prompts.spoon_react import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from spoon_ai.tools import ToolManager

from .toolcall import ToolCallAgent
from .mcp_client_mixin import MCPClientMixin

logger = logging.getLogger(__name__)

class SpoonReactAI(ToolCallAgent):

    name: str = "spoon_react"
    description: str = "A smart ai agent in neo blockchain"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 10
    tool_choice: str = "auto"

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    llm: ChatBot = Field(default_factory=lambda: ChatBot())

    mcp_transport: Union[str, WSTransport, SSETransport, PythonStdioTransport, FastMCPTransport] = Field(default="mcp_server")
    mcp_topics: List[str] = Field(default=["spoon_react"])

    def __init__(self, **kwargs):
        """Initialize SpoonReactAI with both ToolCallAgent and MCPClientMixin initialization"""
        # Call parent class initializers
        ToolCallAgent.__init__(self, **kwargs)
        MCPClientMixin.__init__(self, mcp_transport=kwargs.get('mcp_transport', SSETransport("http://127.0.0.1:8765/sse")))

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
