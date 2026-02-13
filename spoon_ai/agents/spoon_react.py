from typing import List, Union, Any, Dict, Optional
import asyncio
from fastmcp.client.transports import (FastMCPTransport, PythonStdioTransport,
                                       SSETransport, WSTransport, NpxStdioTransport,
                                       FastMCPStdioTransport, UvxStdioTransport, StdioTransport)
from fastmcp.client import Client as MCPClient
from pydantic import Field, AliasChoices, model_validator
import logging

from spoon_ai.chat import ChatBot
from spoon_ai.prompts.spoon_react import NEXT_STEP_PROMPT_TEMPLATE, SYSTEM_PROMPT
from spoon_ai.tools import ToolManager


from .toolcall import ToolCallAgent
from .mcp_client_mixin import MCPClientMixin

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

class SpoonReactAI(MCPClientMixin, ToolCallAgent):

    name: str = "spoon_react"
    description: str = "A smart ai agent in neo blockchain"

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    max_steps: int = 10
    tool_choice: str = "required"

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager([]),
        validation_alias=AliasChoices("available_tools", "avaliable_tools", "tools"),
    )
    llm: ChatBot = Field(default_factory=create_configured_chatbot)

    mcp_transport: Union[str, WSTransport, SSETransport, PythonStdioTransport, NpxStdioTransport, FastMCPTransport, FastMCPStdioTransport, UvxStdioTransport, StdioTransport] = Field(default="mcp_server")
    mcp_topics: List[str] = Field(default=["spoon_react"])
    x402_enabled: bool = Field(default=True, description="Automatically attach x402 payment tools when configuration is present.")

    def __init__(self, **kwargs):
        """Initialize SpoonReactAI with both ToolCallAgent and MCPClientMixin initialization"""
        # Call parent class initializers
        ToolCallAgent.__init__(self, **kwargs)

        # Initialize MCP client mixin
        MCPClientMixin.__init__(self, self.mcp_transport)

        # Normalize available_tools input (list -> ToolManager)
        if isinstance(getattr(self, "available_tools", None), list):
            self.available_tools = ToolManager(self.available_tools)
        if self.available_tools is None:
            self.available_tools = ToolManager([])
        self._x402_tools_initialized = False
        self._refresh_prompts()

    @model_validator(mode="before")
    @classmethod
    def _coerce_tools(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Allow passing `tools` or `available_tools` as a list; wrap into ToolManager."""
        tools_input = values.get("tools", None)
        avail_input = values.get("available_tools", None) or values.get("avaliable_tools", None)

        def wrap(val):
            if isinstance(val, ToolManager):
                return val
            if isinstance(val, list):
                return ToolManager(val)
            return val

        if tools_input is not None:
            values["available_tools"] = wrap(tools_input)
        elif avail_input is not None:
            values["available_tools"] = wrap(avail_input)

        return values

    def _build_tool_list(self) -> str:
        """Return bullet list of available tools names and descriptions."""
        if not getattr(self, "available_tools", None) or not getattr(self.available_tools, "tool_map", None):
            return "- (no tools loaded)"
        lines = []
        for tool in self.available_tools.tool_map.values():
            desc = getattr(tool, "description", "") or ""
            lines.append(f"- {getattr(tool, 'name', 'unknown')}: {desc}")
        return "\n".join(lines)

    def _refresh_prompts(self) -> None:
        """Refresh system and next-step prompts dynamically from current tools."""
        tool_list = self._build_tool_list()
        self.system_prompt = f"{SYSTEM_PROMPT}\n\nAvailable tools:\n{tool_list}"
        self.next_step_prompt = NEXT_STEP_PROMPT_TEMPLATE.format(
            tool_list=tool_list,
        )

    async def connect(self):
        """Establish connection to MCP server."""
        if self.mcp_transport == "mcp_server":
            logger.debug(f"Skipping MCP connection for default 'mcp_server' transport in agent {self.name}")
            return

        try:
            # MCPClientMixin.get_session() handles the actual connection
            async with self.get_session() as session:
                logger.info(f"Successfully connected to MCP server via {self.mcp_transport}")
        except Exception as e:
            logger.warning(f"Failed to connect to MCP server: {e}. Agent will continue without MCP tools.")

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

    async def run(self, request: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """Ensure prompts reflect current tools before running."""
        self._refresh_prompts()
        kwargs: dict = {}
        if request is not None:
            kwargs["request"] = request
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await super().run(**kwargs)
