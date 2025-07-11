from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import SSETransport
from spoon_ai.tools.tool_manager import ToolManager
from pydantic import Field
class SpoonReactMCP(SpoonReactAI, MCPClientMixin):
    name: str = "spoon_react_mcp"
    description: str = "A smart ai agent in neo blockchain with mcp"
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MCPClientMixin.__init__(self, mcp_transport=kwargs.get('mcp_transport', SSETransport("http://127.0.0.1:8765/sse")))
