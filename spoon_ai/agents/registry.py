import importlib
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
from abc import ABC, abstractmethod

from .base import BaseAgent
from .mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import (
    FastMCPTransport, 
    PythonStdioTransport,
    SSETransport, 
    WSTransport,
    StdioTransport
)

logger = logging.getLogger(__name__)


class AgentInterface(ABC):
    """
    Abstract agent interface that defines standard methods all agents must implement
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent
        
        Returns:
            bool: Whether initialization was successful
        """
        pass
    
    @abstractmethod
    async def run(self, request: Optional[str] = None) -> str:
        """
        Run the agent to process a request
        
        Args:
            request: User request
            
        Returns:
            str: Processing result
        """
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools for the agent
        
        Returns:
            List[Dict[str, Any]]: List of tools
        """
        pass


class AgentRegistry:
    """
    Agent registry that supports dynamic loading and management of agents through configuration files
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the agent registry
        
        Args:
            config_path: Configuration file path, supports YAML and JSON formats
        """
        self.config = {}
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_directories: List[Path] = []
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration file
        
        Args:
            config_path: Configuration file path
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            logger.info(f"Successfully loaded configuration file: {config_path}")
            
            # Load plugin directories
            if 'plugin_directories' in self.config:
                self.load_plugin_directories(self.config['plugin_directories'])
                
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            raise
    
    def load_plugin_directories(self, plugin_dirs: List[str]) -> None:
        """
        Load plugin directories
        
        Args:
            plugin_dirs: List of plugin directories
        """
        for plugin_dir in plugin_dirs:
            plugin_path = Path(plugin_dir)
            if plugin_path.exists() and plugin_path.is_dir():
                self.plugin_directories.append(plugin_path)
                logger.info(f"Added plugin directory: {plugin_path}")
            else:
                logger.warning(f"Plugin directory does not exist: {plugin_path}")
    
    def _create_mcp_transport(self, transport_config: Dict[str, Any]) -> Union[FastMCPTransport, None]:
        """
        Create MCP transport object based on configuration
        
        Args:
            transport_config: Transport configuration
            
        Returns:
            MCP transport object
        """
        transport_type = transport_config.get('type')
        
        if transport_type == 'SSETransport':
            return SSETransport(transport_config.get('endpoint'))
        elif transport_type == 'WSTransport':
            return WSTransport(transport_config.get('endpoint'))
        elif transport_type == 'StdioTransport':
            return StdioTransport(
                command=transport_config.get('command'),
                args=transport_config.get('args', []),
                env=transport_config.get('env', {})
            )
        elif transport_type == 'PythonStdioTransport':
            return PythonStdioTransport(
                command=transport_config.get('command'),
                args=transport_config.get('args', []),
                env=transport_config.get('env', {})
            )
        else:
            logger.warning(f"Unknown transport type: {transport_type}")
            return None
    
    def _import_agent_class(self, class_path: str) -> Type[BaseAgent]:
        """
        Dynamically import agent class
        
        Args:
            class_path: Class path in format "module.path.ClassName"
            
        Returns:
            Agent class
        """
        try:
            # First try to import from standard path
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            
            return agent_class
        except (ImportError, AttributeError) as e:
            # If standard import fails, try importing from plugin directories
            logger.warning(f"Standard import failed: {e}, trying plugin directories")
            
            for plugin_dir in self.plugin_directories:
                try:
                    # Add plugin directory to system path
                    import sys
                    if str(plugin_dir) not in sys.path:
                        sys.path.insert(0, str(plugin_dir))
                    
                    module = importlib.import_module(module_name)
                    agent_class = getattr(module, class_name)
                    logger.info(f"Successfully imported from plugin directory: {class_path}")
                    return agent_class
                except (ImportError, AttributeError):
                    continue
            
            raise ImportError(f"Unable to import agent class: {class_path}")
    
    def _create_mcp_agent_class(self, base_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """
        Dynamically create agent class with MCP functionality
        
        Args:
            base_class: Base agent class
            
        Returns:
            Enhanced agent class
        """
        # If already an MCP agent, return directly
        if issubclass(base_class, MCPClientMixin):
            return base_class
        
        # Dynamically create multiple inheritance class
        class MCPEnabledAgent(base_class, MCPClientMixin):
            def __init__(self, mcp_transport=None, **kwargs):
                # Initialize base agent first
                base_class.__init__(self, **kwargs)
                
                # If MCP transport is provided, initialize MCP client
                if mcp_transport:
                    MCPClientMixin.__init__(self, mcp_transport=mcp_transport)
        
        return MCPEnabledAgent
    
    async def load_agents(self) -> None:
        """
        Load all agents according to configuration
        """
        if 'agents' not in self.config:
            logger.warning("No agent configuration found in config file")
            return
        
        for agent_config in self.config['agents']:
            await self.load_agent(agent_config)
    
    async def load_agent(self, agent_config: Dict[str, Any]) -> BaseAgent:
        """
        Load a single agent
        
        Args:
            agent_config: Agent configuration
            
        Returns:
            Created agent instance
        """
        name = agent_config.get('name')
        if not name:
            raise ValueError("Agent configuration missing name")
        
        class_path = agent_config.get('class')
        if not class_path:
            raise ValueError(f"Agent {name} configuration missing class path")
        
        try:
            # Dynamically import agent class
            base_agent_class = self._import_agent_class(class_path)
            
            # Prepare initialization parameters
            init_kwargs = {}
            
            # Add basic configuration
            for key, value in agent_config.items():
                if key not in ['name', 'class', 'mcp_transport', 'endpoint']:
                    init_kwargs[key] = value
            
            # Handle MCP transport configuration
            mcp_transport = None
            if 'mcp_transport' in agent_config or 'endpoint' in agent_config:
                # Simplified configuration: directly specify transport type and endpoint
                if 'mcp_transport' in agent_config:
                    transport_type = agent_config['mcp_transport']
                    endpoint = agent_config.get('endpoint')
                    
                    if transport_type == 'SSETransport' and endpoint:
                        from fastmcp.client.transports import SSETransport
                        mcp_transport = SSETransport(endpoint)
                    elif transport_type == 'WSTransport' and endpoint:
                        from fastmcp.client.transports import WSTransport
                        mcp_transport = WSTransport(endpoint)
                
                # Or complete transport configuration
                elif 'transport_config' in agent_config:
                    mcp_transport = self._create_mcp_transport(agent_config['transport_config'])
            
            # Create enhanced agent class
            agent_class = self._create_mcp_agent_class(base_agent_class)
            
            # Create agent instance
            if mcp_transport:
                agent = agent_class(mcp_transport=mcp_transport, **init_kwargs)
            else:
                agent = agent_class(**init_kwargs)
            
            # Set agent name
            if hasattr(agent, 'name'):
                agent.name = name
            
            # Save agent and configuration
            self.agents[name] = agent
            self.agent_configs[name] = agent_config
            
            logger.info(f"Successfully loaded agent: {name} ({class_path})")
            
            # If agent implements initialize method, call it
            if hasattr(agent, 'initialize'):
                await agent.initialize()
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to load agent {name}: {e}")
            raise
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get agent by name
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance, or None if not found
        """
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """
        List all loaded agent names
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    async def remove_agent(self, name: str) -> bool:
        """
        Remove an agent
        
        Args:
            name: Agent name
            
        Returns:
            Whether removal was successful
        """
        if name in self.agents:
            agent = self.agents[name]
            
            # If agent has cleanup method, call it
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up agent {name}: {e}")
            
            del self.agents[name]
            if name in self.agent_configs:
                del self.agent_configs[name]
            
            logger.info(f"Removed agent: {name}")
            return True
        
        return False
    
    async def cleanup(self) -> None:
        """
        Clean up all agents
        """
        for name in list(self.agents.keys()):
            await self.remove_agent(name)
        
        logger.info("All agents cleaned up") 