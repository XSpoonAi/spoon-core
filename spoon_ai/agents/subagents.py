"""
Subagent Orchestration System

Enables hierarchical agent delegation where a parent agent can create
and manage specialized child agents for complex tasks.

Features:
- Subagent specification and compilation
- State inheritance and isolation
- Hierarchical task delegation with recursion depth limits
- Automatic task tool generation

Usage:
    # Define subagents
    subagents = [
        SubAgentSpec(
            name="researcher",
            description="Specialized in research tasks",
            system_prompt="You are a research expert...",
            tools=[search_tool, summarize_tool]
        )
    ]

    # Create parent agent with subagent support
    agent = ToolCallAgent(...)
    agent.with_subagents(subagents)
"""

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from copy import deepcopy

from spoon_ai.schema import Message, Role
from spoon_ai.tools.base import BaseTool
from spoon_ai.middleware.base import AgentMiddleware, AgentRuntime, ToolCallRequest, ToolCallResult

logger = logging.getLogger(__name__)


# ============================================================================
# Subagent Specification
# ============================================================================

@dataclass
class SubAgentSpec:
    """Specification for a subagent.

    This defines how a subagent should be configured and what
    capabilities it has.
    """
    # Identification
    name: str
    description: str  # For parent agent to decide when to delegate

    # Configuration
    system_prompt: str
    tools: List[BaseTool]

    # Optional overrides
    model: Optional[str] = None
    max_steps: Optional[int] = None
    temperature: Optional[float] = None

    # Middleware for subagent
    middleware: Optional[List[AgentMiddleware]] = None

    # HITL configuration for subagent
    interrupt_on: Optional[Dict[str, Any]] = None


# State keys that should NOT be inherited by subagents
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "plan"}


# ============================================================================
# Subagent Manager
# ============================================================================

class SubAgentManager:
    """Manages subagent creation and task delegation with recursion safety."""

    def __init__(
        self,
        parent_agent: Any,  # BaseAgent type
        subagent_specs: List[SubAgentSpec],
        default_middleware: Optional[List[AgentMiddleware]] = None,
        max_depth: int = 3  
    ):
        """Initialize subagent manager.

        Args:
            parent_agent: The parent agent instance
            subagent_specs: List of subagent specifications
            default_middleware: Default middleware for all subagents
            max_depth: Maximum recursion depth for subagent delegation (default: 3)
        """
        self.parent = parent_agent
        self.subagent_specs = {spec.name: spec for spec in subagent_specs}
        self.default_middleware = default_middleware or []
        self.max_depth = max_depth  

        # Compiled subagent instances (lazy initialization)
        self._subagent_instances: Dict[str, Any] = {}

        logger.info(f"SubAgentManager initialized with {len(subagent_specs)} subagent specs, max_depth={max_depth}")

    def _compile_subagent(self, spec: SubAgentSpec) -> Any:
        """Compile a subagent from specification.

        Args:
            spec: Subagent specification

        Returns:
            Compiled agent instance
        """
        # Import here to avoid circular dependency
        from spoon_ai.agents.toolcall import ToolCallAgent
        from spoon_ai.tools import ToolManager

        # Combine default middleware with subagent-specific middleware
        middleware = self.default_middleware.copy()
        if spec.middleware:
            middleware.extend(spec.middleware)

        # Create ToolManager for subagent
        tool_manager = ToolManager(tools=[])
        if hasattr(spec, 'tools') and spec.tools:
            for tool in spec.tools:
                tool_manager.add_tool(tool)

        if self.parent is None:
            raise RuntimeError(
                f"Cannot compile subagent '{spec.name}': parent agent not set. "
                "This usually means before_agent() hook hasn't been called yet. "
                f"Manager id: {id(self)}"
            )

        # Additional safety check
        if not hasattr(self.parent, 'name'):
            raise RuntimeError(
                f"Cannot compile subagent '{spec.name}': parent agent has no 'name' attribute. "
                f"Parent type: {type(self.parent)}, Parent value: {self.parent}"
            )

        # Create subagent instance
        logger.debug(f"Compiling subagent {spec.name} with parent {self.parent.name}")
        subagent = ToolCallAgent(
            name=f"{self.parent.name}/{spec.name}",
            llm=self.parent.llm,  # Inherit LLM from parent
            system_prompt=spec.system_prompt,
            available_tools=tool_manager,
            max_steps=spec.max_steps or self.parent.max_steps,
            middleware=middleware if middleware else []
        )

        # Override model if specified
        if spec.model:
            # Create new LLM with different model
            from spoon_ai.chat import ChatBot
            subagent.llm = ChatBot(
                provider=self.parent.llm.provider if hasattr(self.parent.llm, 'provider') else "openai",
                model=spec.model,
                temperature=spec.temperature or 0.3
            )

        logger.info(f"Compiled subagent '{spec.name}' with {len(spec.tools)} tools")

        return subagent

    def get_subagent(self, name: str) -> Optional[Any]:
        """Get or create a subagent instance.

        Args:
            name: Subagent name

        Returns:
            Subagent instance or None if not found
        """
        if name not in self.subagent_specs:
            logger.error(f"Subagent '{name}' not found in specifications")
            return None

        # Lazy initialization
        if name not in self._subagent_instances:
            spec = self.subagent_specs[name]
            self._subagent_instances[name] = self._compile_subagent(spec)

        return self._subagent_instances[name]

    async def delegate_task(
        self,
        subagent_name: str,
        task_description: str,
        inherit_state: bool = True
    ) -> str:
        """Delegate a task to a subagent with recursion depth checking.

        Args:
            subagent_name: Name of the subagent
            task_description: Task for the subagent
            inherit_state: Whether to inherit parent state

        Returns:
            Subagent's final response
        """
        depth = int(getattr(self.parent, "_agent_state", {}).get("_subagent_depth", 0))
        if depth >= self.max_depth:
            error_msg = f"Error: Maximum subagent depth {self.max_depth} exceeded. Cannot delegate further."
            logger.error(error_msg)
            return error_msg

        subagent = self.get_subagent(subagent_name)
        if not subagent:
            return f"Error: Subagent '{subagent_name}' not found. Available subagents: {list(self.subagent_specs.keys())}"

        logger.info(f"Delegating task to subagent '{subagent_name}' at depth {depth + 1}/{self.max_depth}: {task_description[:100]}...")

       
        if hasattr(subagent, 'memory'):
            subagent.memory.clear()
        if hasattr(subagent, '_agent_state'):
            # Deep copy to prevent shared mutable objects
            subagent._agent_state = {}
        subagent.current_step = 0
        from spoon_ai.schema import AgentState
        subagent.state = AgentState.IDLE

        # Prepare subagent state with depth tracking
        if inherit_state:
            # Inherit parent state but exclude certain keys
            parent_state = self.parent._agent_state if hasattr(self.parent, '_agent_state') else {}
            
            subagent_state = deepcopy({
                k: v for k, v in parent_state.items()
                if k not in _EXCLUDED_STATE_KEYS
            })

            subagent_state["_subagent_depth"] = depth + 1

            if hasattr(subagent, '_agent_state'):
                subagent._agent_state = subagent_state

        # Run subagent
        try:
            result = await subagent.run(task_description)
            logger.info(f"Subagent '{subagent_name}' completed task at depth {depth + 1}")
            return result

        except Exception as e:
            logger.error(f"Subagent '{subagent_name}' failed at depth {depth + 1}: {e}")
            return f"Error: Subagent execution failed: {str(e)}"

    def create_task_tool(self) -> BaseTool:
        """Create the 'task' tool for delegating to subagents.

        Returns:
            Task delegation tool
        """
        # Create tool description with subagent list
        subagent_descriptions = "\n".join([
            f"  - {name}: {spec.description}"
            for name, spec in self.subagent_specs.items()
        ])

        class TaskTool(BaseTool):
            """Tool for delegating tasks to specialized subagents."""

            name: str = "task"
            description: str = f"""Delegate a task to a specialized subagent.

Use this when you need specialized capabilities or want to break down
complex work into focused subtasks.

Available subagents:
{subagent_descriptions}

Args:
    description: Detailed description of the task
    subagent_type: Name of the subagent to use

Returns:
    Result from the subagent
"""
            parameters: dict = {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed task description for the subagent"
                    },
                    "subagent_type": {
                        "type": "string",
                        "enum": list(self.subagent_specs.keys()),
                        "description": "Name of the subagent to delegate to"
                    }
                },
                "required": ["description", "subagent_type"]
            }

            # Pydantic field for manager reference
            _manager: Any = None

            def __init__(self, manager: 'SubAgentManager', **kwargs):
                super().__init__(**kwargs)
                # Store manager reference (not a Pydantic field)
                object.__setattr__(self, '_manager', manager)

            async def execute(self, description: str, subagent_type: str, **kwargs) -> str:
                """Execute task delegation."""
                return await self._manager.delegate_task(subagent_type, description)

        return TaskTool(self)


# ============================================================================
# Subagent Middleware
# ============================================================================

class SubAgentMiddleware(AgentMiddleware):
    """Middleware that adds subagent orchestration capabilities.

    This middleware:
    1. Injects the 'task' tool for delegation
    2. Manages subagent lifecycle
    3. Handles state inheritance
    4. Enforces recursion depth limits

    Usage:
        middleware = SubAgentMiddleware(subagents=[
            SubAgentSpec(
                name="researcher",
                description="Research and gather information",
                system_prompt="You are a research expert...",
                tools=[search_tool]
            )
        ])

        agent = ToolCallAgent(
            middleware=[middleware],
            ...
        )
    """

    system_prompt = """# Task Delegation

You can delegate specialized tasks to subagents using the 'task' tool.
Each subagent has specific expertise and tools.

When to delegate:
- Task requires specialized knowledge
- Task is complex and can be broken down
- You need focused execution on a subtask

Best practices:
- Provide clear, detailed task descriptions
- Choose the most appropriate subagent
- Review subagent results before proceeding
"""

    def __init__(
        self,
        subagents: List[SubAgentSpec],
        default_middleware: Optional[List[AgentMiddleware]] = None,
        max_depth: int = 3  
    ):
        """Initialize subagent middleware.

        Args:
            subagents: List of subagent specifications
            default_middleware: Default middleware for all subagents
            max_depth: Maximum recursion depth for subagent delegation
        """
        super().__init__()
        self.subagent_specs = subagents
        self.default_middleware = default_middleware

        # Manager will be initialized in before_agent
        self._manager: Optional[SubAgentManager] = None

        
        # so that collect_tools() can find it during pipeline initialization
        self._temp_manager = SubAgentManager(
            parent_agent=None,  # Will be set in before_agent
            subagent_specs=subagents,
            default_middleware=default_middleware,
            max_depth=max_depth  
        )

        # Expose tools property for pipeline collection
        self.tools = [self._temp_manager.create_task_tool()]
        logger.info(f"SubAgentMiddleware initialized with task tool for {len(subagents)} subagents, max_depth={max_depth}")

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Initialize subagent manager with parent agent reference.

        We need access to the parent agent for proper delegation.
        """
        # Get parent agent from runtime if available
        parent_agent = getattr(runtime, '_agent_instance', None)

        logger.info(f"🔍 SubAgentMiddleware.before_agent called. parent_agent={parent_agent}, has _manager={self._manager is not None}")

        if parent_agent and not self._manager:
            # FIXED: Update the temporary manager's parent reference
            self._temp_manager.parent = parent_agent
            self._manager = self._temp_manager
            logger.info(f"✅ Initialized SubAgentManager with parent agent {parent_agent.name} (manager id: {id(self._temp_manager)})")
        elif not parent_agent:
            logger.warning("⚠️ before_agent called but parent_agent is None!")
        elif self._manager:
            logger.info(f"SubAgentManager already initialized (manager id: {id(self._manager)})")

    
        if "_subagent_depth" not in state:
            return {"_subagent_depth": 0}

        return None


# ============================================================================
# Integration Helper for BaseAgent
# ============================================================================

def add_subagent_support(
    agent: Any,
    subagents: List[SubAgentSpec],
    max_depth: int = 3  
) -> Any:
    """Add subagent support to an existing agent.

    This is a helper function that can be used to add subagent capabilities
    to any agent without using middleware.

    Args:
        agent: The agent instance
        subagents: List of subagent specifications
        max_depth: Maximum recursion depth for subagent delegation

    Returns:
        The agent with subagent support

    Example:
        agent = ToolCallAgent(...)
        agent = add_subagent_support(agent, [
            SubAgentSpec(name="researcher", ...)
        ])
    """
    # Create manager with depth limit
    manager = SubAgentManager(agent, subagents, max_depth=max_depth)

    # Add task tool
    task_tool = manager.create_task_tool()
    if hasattr(agent, 'available_tools'):
        agent.available_tools.add_tool(task_tool)

    # Store manager reference
    agent._subagent_manager = manager

    logger.info(f"Added subagent support to agent '{agent.name}' with max_depth={max_depth}")

    return agent


# ============================================================================
# Convenience Methods
# ============================================================================

def create_general_purpose_subagent(
    name: str = "general-purpose",
    description: str = "General-purpose subagent for various tasks",
    tools: Optional[List[BaseTool]] = None
) -> SubAgentSpec:
    """Create a general-purpose subagent specification.

    Args:
        name: Subagent name
        description: Subagent description
        tools: Tools to provide (defaults to parent's tools)

    Returns:
        SubAgentSpec for general-purpose subagent
    """
    return SubAgentSpec(
        name=name,
        description=description,
        system_prompt="""You are a general-purpose assistant subagent.

Your role is to help with tasks delegated by the main agent.
Execute tasks methodically and return clear results.

Guidelines:
- Follow the task description precisely
- Use available tools effectively
- Return concise, actionable results
- Report any errors or blockers clearly
""",
        tools=tools or [],
        max_steps=10
    )
