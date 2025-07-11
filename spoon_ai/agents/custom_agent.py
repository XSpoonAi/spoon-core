import logging
from typing import List, Optional, Dict, Any, Union
import asyncio

from pydantic import Field

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.schema import AgentState
from spoon_ai.tools import BaseTool, ToolManager

logger = logging.getLogger(__name__)

class CustomAgent(ToolCallAgent):
    """
    Custom Agent class allowing users to create their own agents and add custom tools

    Usage:
    Create custom agent and add tools:
       agent = CustomAgent(name="my_agent", description="My custom agent")
       agent.add_tool(MyCustomTool())
       result = await agent.run("Use my custom tool")
    """

    name: str = "custom_agent"
    description: str = "Intelligent agent with customizable tools"

    system_prompt: str = """You are a powerful AI assistant that can use various tools to complete tasks.
You will follow this workflow:
1. Analyze the user's request
2. Determine which tools to use
3. Call the appropriate tools
4. Analyze the tool output
5. Provide a useful response

When you need to use tools, please use the provided tool API. Don't pretend to call non-existent tools.
"""

    next_step_prompt: str = "Please think about what to do next. You can use available tools or directly answer the user's question."

    max_steps: int = 10
    tool_choice: str = "auto"

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    llm: ChatBot = Field(default_factory=lambda: ChatBot())

    # MCP integration configuration
    output_topic: Optional[str] = None
    mcp_enabled: bool = True


    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent

        Args:
            tool: Tool instance to add
        """
        self.avaliable_tools.add_tool(tool)
        logger.info(f"Added tool: {tool.name}")

    def add_tools(self, tools: List[BaseTool]) -> None:
        """
        Add multiple tools to the agent

        Args:
            tools: List of tool instances to add
        """
        for tool in tools:
            self.add_tool(tool)

    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the agent

        Args:
            tool_name: Name of the tool to remove
        """
        self.avaliable_tools.remove_tool(tool_name)
        logger.info(f"Removed tool: {tool_name}")

    def list_tools(self) -> List[str]:
        """
        List all available tools in the agent

        Returns:
            List of tool names
        """
        return [tool.name for tool in self.avaliable_tools.tools]

    async def run(self, request: Optional[str] = None) -> str:
        """
        Run the agent to process a request

        Args:
            request: User request

        Returns:
            Processing result
        """
        if self.state != AgentState.IDLE:
            self.clear()

        return await super().run(request)

    def clear(self):
        """Clear the Agent's state and memory"""
        super().clear()
        self.current_step = 0
        self.state = AgentState.IDLE

        if hasattr(self, "_last_sender"):
            delattr(self, "_last_sender")
        if hasattr(self, "_last_topic"):
            delattr(self, "_last_topic")
        if hasattr(self, "_last_message_id"):
            delattr(self, "_last_message_id")
