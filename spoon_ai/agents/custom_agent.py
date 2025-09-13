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

    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    llm: ChatBot = Field(default_factory=lambda: ChatBot())

    # MCP integration configuration
    output_topic: Optional[str] = None
    mcp_enabled: bool = True

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent with validation.
        
        Args:
            tool: Tool instance to add
            
        Raises:
            ValueError: If tool is invalid or already exists
        """
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Tool must be an instance of BaseTool, got {type(tool)}")
        
        if not hasattr(tool, 'name') or not tool.name:
            raise ValueError("Tool must have a valid name")
        
        # Check for duplicate tool names
        existing_names = self.list_tools()
        if tool.name in existing_names:
            logger.warning(f"Tool '{tool.name}' already exists, replacing it")
            self.remove_tool(tool.name)
        
        try:
            # ToolManager uses tools list, not add_tool method
            if not hasattr(self.avaliable_tools, 'tools'):
                self.avaliable_tools.tools = []
            
            self.avaliable_tools.tools.append(tool)
            
            # Update tool_map if it exists
            if hasattr(self.avaliable_tools, 'tool_map'):
                self.avaliable_tools.tool_map[tool.name] = tool
            
            logger.info(f"Added tool: {tool.name} (total: {len(self.avaliable_tools.tools)})")
            
        except Exception as e:
            logger.error(f"Failed to add tool {tool.name}: {e}")
            raise ValueError(f"Failed to add tool: {e}") from e

    def add_tools(self, tools: List[BaseTool]) -> None:
        """
        Add multiple tools to the agent with atomic operation.
        
        Args:
            tools: List of tool instances to add
            
        Raises:
            ValueError: If any tool is invalid
        """
        if not isinstance(tools, list):
            raise ValueError("Tools must be provided as a list")
        
        if not tools:
            logger.warning("No tools provided to add")
            return
        
        # Validate all tools first before adding any
        for i, tool in enumerate(tools):
            if not isinstance(tool, BaseTool):
                raise ValueError(f"Tool at index {i} must be an instance of BaseTool, got {type(tool)}")
            if not hasattr(tool, 'name') or not tool.name:
                raise ValueError(f"Tool at index {i} must have a valid name")
        
        # Add all tools (will handle duplicates individually)
        added_count = 0
        for tool in tools:
            try:
                self.add_tool(tool)
                added_count += 1
            except Exception as e:
                logger.error(f"Failed to add tool {getattr(tool, 'name', 'unknown')}: {e}")
                # Continue adding other tools even if one fails
        
        logger.info(f"Added {added_count}/{len(tools)} tools successfully")

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            bool: True if tool was removed, False if not found
        """
        if not isinstance(tool_name, str) or not tool_name.strip():
            logger.error("Tool name must be a non-empty string")
            return False
        
        try:
            # Find and remove from tools list
            if hasattr(self.avaliable_tools, 'tools'):
                original_count = len(self.avaliable_tools.tools)
                self.avaliable_tools.tools = [
                    tool for tool in self.avaliable_tools.tools 
                    if getattr(tool, 'name', '') != tool_name
                ]
                removed = len(self.avaliable_tools.tools) < original_count
            else:
                removed = False
            
            # Remove from tool_map if it exists
            if hasattr(self.avaliable_tools, 'tool_map') and tool_name in self.avaliable_tools.tool_map:
                del self.avaliable_tools.tool_map[tool_name]
                removed = True
            
            if removed:
                logger.info(f"Removed tool: {tool_name} (remaining: {len(self.avaliable_tools.tools)})")
                return True
            else:
                logger.warning(f"Tool '{tool_name}' not found for removal")
                return False
                
        except Exception as e:
            logger.error(f"Error removing tool {tool_name}: {e}")
            return False

    def list_tools(self) -> List[str]:
        """
        List all available tools in the agent.
        
        Returns:
            List of tool names, empty list if no tools
        """
        try:
            if hasattr(self.avaliable_tools, 'tools') and self.avaliable_tools.tools:
                return [getattr(tool, 'name', 'unnamed') for tool in self.avaliable_tools.tools]
            return []
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    def get_tool_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all tools.
        
        Returns:
            Dictionary with tool names as keys and tool info as values
        """
        tool_info = {}
        try:
            if hasattr(self.avaliable_tools, 'tools'):
                for tool in self.avaliable_tools.tools:
                    if hasattr(tool, 'name'):
                        tool_info[tool.name] = {
                            'description': getattr(tool, 'description', 'No description'),
                            'parameters': getattr(tool, 'parameters', {}),
                            'type': type(tool).__name__
                        }
        except Exception as e:
            logger.error(f"Error getting tool info: {e}")
        
        return tool_info
    

    def validate_tools(self) -> Dict[str, Any]:
        """
        Validate all current tools and return validation report.
        
        Returns:
            Dictionary with validation results
        """
        report = {
            'valid_tools': [],
            'invalid_tools': [],
            'total_tools': 0,
            'issues': []
        }
        
        try:
            if not hasattr(self.avaliable_tools, 'tools'):
                report['issues'].append("ToolManager missing 'tools' attribute")
                return report
            
            report['total_tools'] = len(self.avaliable_tools.tools)
            
            for tool in self.avaliable_tools.tools:
                tool_name = getattr(tool, 'name', 'unnamed')
                
                # Basic validation
                if not isinstance(tool, BaseTool):
                    report['invalid_tools'].append(tool_name)
                    report['issues'].append(f"Tool '{tool_name}' is not a BaseTool instance")
                    continue
                
                if not hasattr(tool, 'name') or not tool.name:
                    report['invalid_tools'].append('unnamed')
                    report['issues'].append("Found tool without name")
                    continue
                
                if not hasattr(tool, 'execute'):
                    report['invalid_tools'].append(tool_name)
                    report['issues'].append(f"Tool '{tool_name}' missing execute method")
                    continue
                
                report['valid_tools'].append(tool_name)
            
            logger.info(f"Tool validation: {len(report['valid_tools'])}/{report['total_tools']} tools valid")
            
        except Exception as e:
            report['issues'].append(f"Validation error: {e}")
            logger.error(f"Error during tool validation: {e}")
        
        return report

    async def run(self, request: Optional[str] = None) -> str:
        """
        Run the agent with enhanced tool validation.
        
        Args:
            request: User request
            
        Returns:
            Processing result
        """
        # Validate tools before running
        validation = self.validate_tools()
        if validation['invalid_tools']:
            logger.error(f"Cannot run agent with invalid tools: {validation['invalid_tools']}")
            raise RuntimeError(f"Agent has invalid tools: {', '.join(validation['invalid_tools'])}")
        
        if self.state != AgentState.IDLE:
            self.clear()
        
        return await super().run(request)

    def clear(self):
        """Enhanced clear method with proper tool state management."""
        # Call parent clear first
        super().clear()
        
        # Reset agent-specific state
        self.current_step = 0
        self.state = AgentState.IDLE
        
        # Clean up MCP-related attributes if they exist
        mcp_attributes = ['_last_sender', '_last_topic', '_last_message_id', 'output_topic']
        for attr in mcp_attributes:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Validate tool manager state after clear
        try:
            if hasattr(self.avaliable_tools, 'tools'):
                logger.debug(f"Agent cleared with {len(self.avaliable_tools.tools)} tools remaining")
            else:
                logger.warning("ToolManager missing 'tools' attribute after clear")
        except Exception as e:
            logger.error(f"Error checking tools after clear: {e}")
        
        logger.debug(f"CustomAgent '{self.name}' fully cleared and validated")

    
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