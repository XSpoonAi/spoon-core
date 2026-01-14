# Existing agent imports
from .spoon_react import SpoonReactAI
from .toolcall import ToolCallAgent
from .spoon_react_mcp import SpoonReactMCP

# New agent registry system imports
from .registry import AgentRegistry, AgentInterface
from .enhanced_base import EnhancedBaseAgent, StandardAgent
from .github_agent import GitHubAgent

# Export all available agents and registry-related classes
__all__ = [
    # Original agents
    'SpoonReactAI',
    'ToolCallAgent', 
    'SpoonReactMCP',
    
    # Agent registry system
    'AgentRegistry',
    'AgentInterface',
    'EnhancedBaseAgent',
    'StandardAgent',
    
    # Specific agent implementations
    'GitHubAgent',
]