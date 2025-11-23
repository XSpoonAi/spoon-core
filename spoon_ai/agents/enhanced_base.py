import logging
from typing import List, Dict, Any, Optional
from abc import abstractmethod

from .base import BaseAgent
from .registry import AgentInterface

logger = logging.getLogger(__name__)


class EnhancedBaseAgent(BaseAgent, AgentInterface):
    """
    Enhanced base agent class that inherits from BaseAgent and implements AgentInterface
    Provides standardized agent interface implementation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False
        self._tools = []
    
    async def initialize(self) -> bool:
        """
        Initialize the agent
        
        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Subclasses can override this method to execute specific initialization logic
            await self._initialize_agent()
            self._initialized = True
            logger.info(f"Agent {self.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Agent {self.name} initialization failed: {e}")
            return False
    
    async def _initialize_agent(self) -> None:
        """
        Subclasses can override this method to implement specific initialization logic
        """
        pass
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools for the agent
        
        Returns:
            List[Dict[str, Any]]: List of tools
        """
        tools = []
        
        # If has MCP client, get MCP tools
        if hasattr(self, 'list_mcp_tools'):
            try:
                mcp_tools = await self.list_mcp_tools()
                for tool in mcp_tools:
                    tools.append({
                        'name': tool.name,
                        'description': tool.description,
                        'source': 'mcp',
                        'schema': tool.inputSchema if hasattr(tool, 'inputSchema') else None
                    })
            except Exception as e:
                logger.warning(f"Failed to get MCP tools: {e}")
        
        # If has local tool manager, get local tools
        if hasattr(self, 'avaliable_tools') and self.avaliable_tools:
            try:
                local_tools = self.avaliable_tools.get_tools()
                for tool in local_tools:
                    tools.append({
                        'name': tool.name,
                        'description': tool.description,
                        'source': 'local',
                        'schema': tool.args_schema if hasattr(tool, 'args_schema') else None
                    })
            except Exception as e:
                logger.warning(f"Failed to get local tools: {e}")
        
        self._tools = tools
        return tools
    
    def is_initialized(self) -> bool:
        """
        Check if agent is initialized
        
        Returns:
            bool: Whether agent is initialized
        """
        return self._initialized
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capability description
        
        Returns:
            Dict[str, Any]: Capability description
        """
        capabilities = {
            'name': self.name,
            'description': self.description,
            'max_steps': self.max_steps,
            'has_mcp': hasattr(self, 'list_mcp_tools'),
            'has_local_tools': hasattr(self, 'avaliable_tools'),
            'tools_count': len(await self.list_tools()),
            'initialized': self.is_initialized()
        }
        
        return capabilities
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check
        
        Returns:
            Dict[str, Any]: Health status
        """
        health = {
            'status': 'healthy',
            'initialized': self.is_initialized(),
            'agent_name': self.name,
            'state': self.state.value if hasattr(self.state, 'value') else str(self.state),
            'current_step': self.current_step,
            'max_steps': self.max_steps
        }
        
        # Check MCP connection
        if hasattr(self, 'list_mcp_tools'):
            try:
                await self.list_mcp_tools()
                health['mcp_status'] = 'connected'
            except Exception as e:
                health['mcp_status'] = 'disconnected'
                health['mcp_error'] = str(e)
                health['status'] = 'degraded'
        
        return health
    
    def reset_state(self) -> None:
        """
        Reset agent state
        """
        from spoon_ai.schema import AgentState
        self.state = AgentState.IDLE
        self.current_step = 0
        logger.info(f"Agent {self.name} state reset")
    
    async def cleanup(self) -> None:
        """
        Clean up agent resources
        """
        try:
            # If has MCP client, execute cleanup
            if hasattr(self, 'cleanup') and callable(getattr(self, 'cleanup', None)):
                await super().cleanup()
            
            # Reset state
            self.reset_state()
            self._initialized = False
            
            logger.info(f"Agent {self.name} cleanup completed")
        except Exception as e:
            logger.error(f"Error cleaning up agent {self.name}: {e}")


class StandardAgent(EnhancedBaseAgent):
    """
    Standard agent implementation providing basic agent functionality
    Can be used as a template for creating new agents
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.name:
            self.name = "StandardAgent"
        if not self.description:
            self.description = "Standard agent implementation providing basic conversation and tool calling functionality"
    
    async def step(self) -> str:
        """
        Execute one step
        
        Returns:
            str: Step result
        """
        # Get the last user message
        messages = self.memory.get_messages()
        if not messages:
            return "No messages to process"
        
        last_message = messages[-1]
        if last_message.role.value != "user":
            return "Waiting for user input"
        
        # Use LLM to generate response
        try:
            response = await self.llm.achat(
                messages=[{"role": "user", "content": last_message.content}],
                system_prompt=self.system_prompt or "You are a helpful AI assistant."
            )
            
            # Add assistant response to memory
            self.add_message("assistant", response)
            
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg 