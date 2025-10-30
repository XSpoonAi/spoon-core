from typing import List, Union, Any, Dict, Optional
import asyncio
from fastmcp.client.transports import (FastMCPTransport, PythonStdioTransport,
                                       SSETransport, WSTransport, NpxStdioTransport,
                                       FastMCPStdioTransport, UvxStdioTransport, StdioTransport)
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

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    llm: ChatBot = Field(default_factory=create_configured_chatbot)

    mcp_transport: Union[str, WSTransport, SSETransport, PythonStdioTransport, NpxStdioTransport, FastMCPTransport, FastMCPStdioTransport, UvxStdioTransport, StdioTransport] = Field(default="mcp_server")
    mcp_topics: List[str] = Field(default=["spoon_react"])
    x402_enabled: bool = Field(default=True, description="Automatically attach x402 payment tools when configuration is present.")

    def __init__(self, **kwargs):
        """Initialize SpoonReactAI with both ToolCallAgent and MCPClientMixin initialization"""
        # Call parent class initializers
        ToolCallAgent.__init__(self, **kwargs)
        self._x402_tools_initialized = False

    async def initialize(self, __context: Any = None):
        """Initialize async components and subscribe to topics"""
        logger.info(f"Initializing SpoonReactAI agent '{self.name}'")

        # First establish connection to MCP server
        try:
            # Verify connection
            await self.connect()
            await self._ensure_x402_tools()

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {str(e)}")
            # If context has error handling, use it
            if __context and hasattr(__context, 'report_error'):
                await __context.report_error(e)
            raise

    async def _ensure_x402_tools(self) -> None:
        """Attach x402 helper tools if configuration is available."""
        if not self.x402_enabled or getattr(self, "_x402_tools_initialized", False):
            return

        try:
            from spoon_ai.payments import X402PaymentService, X402ConfigurationError
            from spoon_ai.tools.x402_payment import X402PaymentHeaderTool, X402PaywalledRequestTool
        except ImportError as exc:
            logger.debug(f"x402 integration unavailable: {exc}")
            self._x402_tools_initialized = True
            return

        try:
            service = X402PaymentService()
        except X402ConfigurationError as exc:
            logger.info(f"x402 configuration incomplete, skipping tool attachment: {exc}")
            self._x402_tools_initialized = True
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Unexpected error initialising x402 service: {exc}")
            self._x402_tools_initialized = True
            return

        if getattr(self, "avaliable_tools", None) is None:
            self.avaliable_tools = ToolManager([])

        existing = set(self.avaliable_tools.tool_map.keys())
        if "x402_create_payment" not in existing:
            self.avaliable_tools.add_tool(X402PaymentHeaderTool(service=service))
        if "x402_paywalled_request" not in existing:
            self.avaliable_tools.add_tool(X402PaywalledRequestTool(service=service))

        self._x402_tools_initialized = True
