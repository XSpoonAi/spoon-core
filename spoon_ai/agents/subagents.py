"""
Subagent Orchestration System

Enables hierarchical agent delegation where a parent agent can create
and manage specialized child agents for complex tasks.

Compatible with LangChain DeepAgents SubAgentMiddleware interface.

Features:
- SubAgentSpec for defining subagents with tools and prompts
- CompiledSubAgent for using pre-built graphs as subagents
- Command return pattern for state updates
- State inheritance and isolation
- Hierarchical task delegation with recursion depth limits
- Automatic task tool generation
- General-purpose agent support

Usage:
    # Method 1: SubAgentSpec (simple cases)
    subagents = [
        SubAgentSpec(
            name="researcher",
            description="Specialized in research tasks",
            system_prompt="You are a research expert...",
            tools=[search_tool, summarize_tool]
        )
    ]

    # Method 2: CompiledSubAgent (complex cases with pre-built graph)
    custom_graph = StateGraph(...)
    custom_graph.add_node("analyze", analyze_node)
    compiled = custom_graph.compile()

    subagents = [
        CompiledSubAgent(
            name="complex_analyzer",
            description="Complex multi-step analysis",
            runnable=compiled
        )
    ]

    # Create parent agent with subagent support
    middleware = SubAgentMiddleware(subagents=subagents)
    agent = ToolCallAgent(middleware=[middleware], ...)
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable, Sequence
from dataclasses import dataclass, field
from copy import deepcopy

from spoon_ai.schema import Message, Role
from spoon_ai.tools.base import BaseTool
from spoon_ai.middleware.base import AgentMiddleware, AgentRuntime, ToolCallRequest, ToolCallResult

# Import HITL types for interrupt_on configuration
from spoon_ai.tools.hitl import InterruptOnConfig, HumanInTheLoopMiddleware

logger = logging.getLogger(__name__)


# ============================================================================
# Command Class for State Updates
# ============================================================================

@dataclass
class Command:
    """Command object for returning state updates and controlling execution flow.

    Compatible with LangGraph Command interface. Used to:
    1. Propagate state changes from subagent execution back to the parent agent
    2. Resume from HITL interrupts with approval decisions

    Example - State updates from subagent:
        ```python
        return Command(
            update={
                "messages": [tool_message],
                "some_state_key": new_value,
            }
        )
        ```

    Example - Resume from HITL interrupt:
        ```python
        # After receiving interrupt with action_requests
        result = agent.invoke(
            Command(resume={
                "decisions": [
                    {"type": "approve"},
                    {"type": "edit", "args": {"path": "/new/path"}},
                    {"type": "reject", "reason": "Too dangerous"},
                ]
            }),
            config=config
        )
        ```
    """
    update: Dict[str, Any] = field(default_factory=dict)
    """State updates to apply after subagent execution."""

    goto: Optional[str] = None
    """Optional node to go to next (for graph-based agents)."""

    resume: Optional[Dict[str, Any]] = None
    """Resume data for HITL interrupts. Contains 'decisions' list."""

    def __post_init__(self):
        if self.update is None:
            self.update = {}

    @property
    def is_resume(self) -> bool:
        """Check if this is a resume command."""
        return self.resume is not None

    def get_decisions(self) -> List[Dict[str, Any]]:
        """Get decisions from resume data.

        Returns:
            List of decision dicts with 'type', optional 'args', optional 'reason'
        """
        if not self.resume:
            return []
        return self.resume.get("decisions", [])


# ============================================================================
# Subagent Specifications
# ============================================================================

@dataclass
class SubAgentSpec:
    """Specification for a subagent.

    This defines how a subagent should be configured and what
    capabilities it has. When using SubAgentSpec, the middleware
    will automatically create a ToolCallAgent instance.

    Compatible with LangChain DeepAgents SubAgent interface.
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

    # HITL configuration for subagent (overrides default_interrupt_on)
    interrupt_on: Optional[Dict[str, Union[bool, InterruptOnConfig]]] = None


@dataclass
class CompiledSubAgent:
    """A pre-compiled agent/graph specification.

    Use this when you have a complex, pre-built graph (StateGraph/CompiledGraph)
    that you want to use as a subagent. This is useful for:
    - Complex multi-step workflows
    - Custom graph topologies
    - Reusing existing graph implementations

    Compatible with LangChain DeepAgents CompiledSubAgent interface.

    Example:
        ```python
        from spoon_ai.graph import StateGraph

        # Build custom graph
        graph = StateGraph(MyState)
        graph.add_node("analyze", analyze_node)
        graph.add_node("synthesize", synthesize_node)
        graph.add_edge("analyze", "synthesize")
        compiled = graph.compile()

        # Use as subagent
        subagent = CompiledSubAgent(
            name="analyzer",
            description="Complex multi-step analysis",
            runnable=compiled
        )
        ```
    """
    name: str
    description: str
    runnable: Any  # CompiledGraph or any Runnable-like object with invoke/ainvoke


# Type alias for subagent specifications
SubAgentType = Union[SubAgentSpec, CompiledSubAgent]


# ============================================================================
# Constants and Prompts
# ============================================================================

# State keys that should NOT be inherited by subagents
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "plan"}

# Default description for general-purpose agent
DEFAULT_GENERAL_PURPOSE_DESCRIPTION = (
    "General-purpose agent for researching complex questions, searching for files "
    "and content, and executing multi-step tasks. When you are searching for a "
    "keyword or file and are not confident that you will find the right match in "
    "the first few tries, use this agent to perform the search for you. This agent "
    "has access to all tools as the main agent."
)

DEFAULT_SUBAGENT_PROMPT = (
    "In order to complete the objective that the user asks of you, "
    "you have access to a number of standard tools."
)

# Detailed task tool description with examples - Compatible with LangChain DeepAgents
TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task`tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""

# System prompt for task tool usage
TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""


# ============================================================================
# Subagent Manager
# ============================================================================

class SubAgentManager:
    """Manages subagent creation and task delegation with recursion safety.

    Supports both SubAgentSpec (auto-compiled) and CompiledSubAgent (pre-built).
    Returns Command objects for state updates.
    Compatible with LangChain DeepAgents subagent management.
    """

    def __init__(
        self,
        parent_agent: Any,  # BaseAgent type
        subagent_specs: List[SubAgentType],
        default_middleware: Optional[List[AgentMiddleware]] = None,
        default_tools: Optional[List[BaseTool]] = None,
        default_interrupt_on: Optional[Dict[str, Union[bool, InterruptOnConfig]]] = None,
        max_depth: int = 3,
        general_purpose_agent: bool = True,
    ):
        """Initialize subagent manager.

        Args:
            parent_agent: The parent agent instance
            subagent_specs: List of subagent specifications (SubAgentSpec or CompiledSubAgent)
            default_middleware: Default middleware for all subagents
            default_tools: Default tools for general-purpose agent
            default_interrupt_on: Default HITL configuration for all subagents.
                This is also the fallback for any subagents that don't specify
                their own interrupt_on configuration.
            max_depth: Maximum recursion depth for subagent delegation (default: 3)
            general_purpose_agent: Whether to include a general-purpose subagent (default: True)
        """
        self.parent = parent_agent
        self.default_middleware = default_middleware or []
        self.default_tools = default_tools or []
        self.default_interrupt_on = default_interrupt_on
        self.max_depth = max_depth
        self.general_purpose_agent = general_purpose_agent

        # Separate specs by type
        self.subagent_specs: Dict[str, SubAgentSpec] = {}
        self.compiled_subagents: Dict[str, CompiledSubAgent] = {}

        for spec in subagent_specs:
            if isinstance(spec, CompiledSubAgent):
                self.compiled_subagents[spec.name] = spec
            else:
                self.subagent_specs[spec.name] = spec

        # Compiled subagent instances (lazy initialization for SubAgentSpec)
        self._subagent_instances: Dict[str, Any] = {}

        # Pre-register compiled subagents
        for name, compiled in self.compiled_subagents.items():
            self._subagent_instances[name] = compiled.runnable

        logger.info(
            f"SubAgentManager initialized: {len(self.subagent_specs)} specs, "
            f"{len(self.compiled_subagents)} compiled, max_depth={max_depth}, "
            f"general_purpose={general_purpose_agent}, "
            f"has_interrupt_on={default_interrupt_on is not None}"
        )

    def _get_all_subagent_names(self) -> List[str]:
        """Get all available subagent names."""
        names = list(self.subagent_specs.keys()) + list(self.compiled_subagents.keys())
        if self.general_purpose_agent and "general-purpose" not in names:
            names.insert(0, "general-purpose")
        return names

    def _get_subagent_descriptions(self) -> List[str]:
        """Get formatted descriptions for all subagents."""
        descriptions = []

        # General-purpose agent first
        if self.general_purpose_agent:
            descriptions.append(f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")

        # SubAgentSpec descriptions
        for name, spec in self.subagent_specs.items():
            descriptions.append(f"- {name}: {spec.description}")

        # CompiledSubAgent descriptions
        for name, compiled in self.compiled_subagents.items():
            descriptions.append(f"- {name}: {compiled.description}")

        return descriptions

    def _compile_subagent(self, spec: SubAgentSpec) -> Any:
        """Compile a subagent from specification.

        Includes HITL middleware if interrupt_on is configured (either from spec
        or from default_interrupt_on).
        """
        from spoon_ai.agents.toolcall import ToolCallAgent
        from spoon_ai.tools import ToolManager

        middleware = self.default_middleware.copy()
        if spec.middleware:
            middleware.extend(spec.middleware)

        # Add HITL middleware if interrupt_on is configured
        # Use spec's interrupt_on if provided, otherwise fall back to default_interrupt_on
        interrupt_on = spec.interrupt_on if spec.interrupt_on is not None else self.default_interrupt_on
        if interrupt_on:
            middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
            logger.debug(f"Added HITL middleware to subagent '{spec.name}'")

        tool_manager = ToolManager(tools=[])
        if hasattr(spec, 'tools') and spec.tools:
            for tool in spec.tools:
                tool_manager.add_tool(tool)

        if self.parent is None:
            raise RuntimeError(
                f"Cannot compile subagent '{spec.name}': parent agent not set."
            )

        if not hasattr(self.parent, 'name'):
            raise RuntimeError(
                f"Cannot compile subagent '{spec.name}': parent agent has no 'name' attribute."
            )

        logger.debug(f"Compiling subagent {spec.name} with parent {self.parent.name}")
        subagent = ToolCallAgent(
            name=f"{self.parent.name}/{spec.name}",
            llm=self.parent.llm,
            system_prompt=spec.system_prompt,
            available_tools=tool_manager,
            max_steps=spec.max_steps or self.parent.max_steps,
            middleware=middleware if middleware else []
        )

        if spec.model:
            from spoon_ai.chat import ChatBot
            subagent.llm = ChatBot(
                provider=self.parent.llm.provider if hasattr(self.parent.llm, 'provider') else "openai",
                model=spec.model,
                temperature=spec.temperature or 0.3
            )

        logger.info(f"Compiled subagent '{spec.name}' with {len(spec.tools)} tools")
        return subagent

    def _compile_general_purpose_agent(self) -> Any:
        """Compile the general-purpose agent.

        Includes HITL middleware if default_interrupt_on is configured.
        """
        from spoon_ai.agents.toolcall import ToolCallAgent
        from spoon_ai.tools import ToolManager

        if self.parent is None:
            raise RuntimeError("Cannot compile general-purpose agent: parent agent not set.")

        tools = self.default_tools
        if not tools and hasattr(self.parent, 'available_tools'):
            tools = list(self.parent.available_tools.get_tools().values())

        tool_manager = ToolManager(tools=[])
        for tool in tools:
            tool_manager.add_tool(tool)

        # Build middleware list with optional HITL
        middleware = self.default_middleware.copy()
        if self.default_interrupt_on:
            middleware.append(HumanInTheLoopMiddleware(interrupt_on=self.default_interrupt_on))
            logger.debug("Added HITL middleware to general-purpose subagent")

        subagent = ToolCallAgent(
            name=f"{self.parent.name}/general-purpose",
            llm=self.parent.llm,
            system_prompt=DEFAULT_SUBAGENT_PROMPT,
            available_tools=tool_manager,
            max_steps=self.parent.max_steps,
            middleware=middleware
        )

        logger.info(f"Compiled general-purpose subagent with {len(tools)} tools")
        return subagent

    def get_subagent(self, name: str) -> Optional[Any]:
        """Get or create a subagent instance."""
        if name in self._subagent_instances:
            return self._subagent_instances[name]

        if name == "general-purpose" and self.general_purpose_agent:
            self._subagent_instances[name] = self._compile_general_purpose_agent()
            return self._subagent_instances[name]

        if name in self.subagent_specs:
            spec = self.subagent_specs[name]
            self._subagent_instances[name] = self._compile_subagent(spec)
            return self._subagent_instances[name]

        logger.error(f"Subagent '{name}' not found in specifications")
        return None

    def _return_command_with_state_update(
        self,
        result: Dict[str, Any],
        tool_call_id: str,
        final_message: str
    ) -> Command:
        """Create a Command with state updates from subagent result.

        Compatible with LangChain's _return_command_with_state_update pattern.
        """
        # Extract state updates, excluding certain keys
        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}

        # Strip trailing whitespace to prevent API errors with Anthropic
        message_text = final_message.rstrip() if final_message else ""

        # Create tool message for the response
        tool_message = Message(
            role=Role.TOOL,
            content=message_text,
            tool_call_id=tool_call_id,
        )

        return Command(
            update={
                **state_update,
                "messages": [tool_message],
            }
        )

    async def delegate_task(
        self,
        subagent_name: str,
        task_description: str,
        inherit_state: bool = True,
        tool_call_id: Optional[str] = None,
    ) -> Union[str, Command]:
        """Delegate a task to a subagent with recursion depth checking.

        Args:
            subagent_name: Name of the subagent
            task_description: Task for the subagent
            inherit_state: Whether to inherit parent state
            tool_call_id: Tool call ID for Command return pattern

        Returns:
            Subagent's final response as string, or Command with state updates
        """
        # Check recursion depth
        depth = int(getattr(self.parent, "_agent_state", {}).get("_subagent_depth", 0))
        if depth >= self.max_depth:
            error_msg = f"Error: Maximum subagent depth {self.max_depth} exceeded. Cannot delegate further."
            logger.error(error_msg)
            return error_msg

        # Validate subagent exists
        available = self._get_all_subagent_names()
        if subagent_name not in available:
            allowed_types = ", ".join([f"`{k}`" for k in available])
            return f"We cannot invoke subagent {subagent_name} because it does not exist, the only allowed types are {allowed_types}"

        # Get subagent
        subagent = self.get_subagent(subagent_name)
        if not subagent:
            return f"Error: Failed to compile subagent '{subagent_name}'"

        logger.info(
            f"Delegating task to subagent '{subagent_name}' at depth {depth + 1}/{self.max_depth}: "
            f"{task_description[:100]}..."
        )

        # Prepare subagent state
        subagent_state = {"_subagent_depth": depth + 1}

        if inherit_state and hasattr(self.parent, '_agent_state'):
            parent_state = self.parent._agent_state
            subagent_state.update({
                k: deepcopy(v) for k, v in parent_state.items()
                if k not in _EXCLUDED_STATE_KEYS
            })

        # Run subagent
        try:
            # Check if it's a CompiledSubAgent (has invoke/ainvoke)
            if hasattr(subagent, 'ainvoke'):
                # It's a compiled graph/runnable
                input_state = {
                    "messages": [Message(role=Role.USER, content=task_description)],
                    **subagent_state
                }
                result = await subagent.ainvoke(input_state)

                # Extract final message from result
                final_message = ""
                if isinstance(result, dict) and "messages" in result:
                    messages = result["messages"]
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, 'content'):
                            final_message = str(last_msg.content)
                        elif hasattr(last_msg, 'text'):
                            final_message = str(last_msg.text)
                        else:
                            final_message = str(last_msg)

                # Return Command if tool_call_id provided
                if tool_call_id:
                    return self._return_command_with_state_update(
                        result if isinstance(result, dict) else {},
                        tool_call_id,
                        final_message
                    )
                return final_message

            else:
                # It's a regular agent (ToolCallAgent)
                if hasattr(subagent, 'memory'):
                    subagent.memory.clear()
                if hasattr(subagent, '_agent_state'):
                    subagent._agent_state = subagent_state
                subagent.current_step = 0
                from spoon_ai.schema import AgentState
                subagent.state = AgentState.IDLE

                result = await subagent.run(task_description)
                logger.info(f"Subagent '{subagent_name}' completed task at depth {depth + 1}")

                # Return Command if tool_call_id provided
                if tool_call_id:
                    return self._return_command_with_state_update(
                        subagent_state,
                        tool_call_id,
                        result
                    )
                return result

        except Exception as e:
            logger.error(f"Subagent '{subagent_name}' failed at depth {depth + 1}: {e}")
            error_msg = f"Error: Subagent execution failed: {str(e)}"
            if tool_call_id:
                return self._return_command_with_state_update({}, tool_call_id, error_msg)
            return error_msg

    def create_task_tool(self, task_description: Optional[str] = None) -> BaseTool:
        """Create the 'task' tool for delegating to subagents.

        Args:
            task_description: Custom description for the task tool.
                Supports {available_agents} placeholder.

        Returns:
            Task delegation tool
        """
        # Get subagent descriptions
        subagent_descriptions = self._get_subagent_descriptions()
        subagent_description_str = "\n".join(subagent_descriptions)

        # Format task description - compatible with LangChain pattern
        if task_description is None:
            formatted_description = TASK_TOOL_DESCRIPTION.format(
                available_agents=subagent_description_str
            )
        elif "{available_agents}" in task_description:
            # If custom description has placeholder, format with agent descriptions
            formatted_description = task_description.format(
                available_agents=subagent_description_str
            )
        else:
            formatted_description = task_description

        # Get available subagent names
        available_names = self._get_all_subagent_names()

        class TaskTool(BaseTool):
            """Tool for delegating tasks to specialized subagents."""

            name: str = "task"
            description: str = formatted_description
            parameters: dict = {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed task description for the subagent"
                    },
                    "subagent_type": {
                        "type": "string",
                        "enum": available_names,
                        "description": "Name of the subagent to delegate to"
                    }
                },
                "required": ["description", "subagent_type"]
            }

            _manager: Any = None

            def __init__(self, manager: 'SubAgentManager', **kwargs):
                super().__init__(**kwargs)
                object.__setattr__(self, '_manager', manager)

            async def execute(
                self,
                description: str,
                subagent_type: str,
                tool_call_id: Optional[str] = None,
                **kwargs
            ) -> Union[str, Command]:
                """Execute task delegation.

                Returns either a string result or a Command with state updates.
                """
                return await self._manager.delegate_task(
                    subagent_type,
                    description,
                    tool_call_id=tool_call_id
                )

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
    5. Returns Command objects for state updates
    6. Propagates HITL configuration to subagents

    Supports both SubAgentSpec (auto-compiled) and CompiledSubAgent (pre-built graphs).

    Compatible with LangChain DeepAgents SubAgentMiddleware interface.

    Usage:
        ```python
        # Method 1: SubAgentSpec (simple cases)
        middleware = SubAgentMiddleware(subagents=[
            SubAgentSpec(
                name="researcher",
                description="Research and gather information",
                system_prompt="You are a research expert...",
                tools=[search_tool]
            )
        ])

        # Method 2: CompiledSubAgent (complex cases)
        custom_graph = StateGraph(...)
        compiled = custom_graph.compile()

        middleware = SubAgentMiddleware(subagents=[
            CompiledSubAgent(
                name="analyzer",
                description="Complex analysis workflow",
                runnable=compiled
            )
        ])

        # Method 3: With HITL configuration inherited by subagents
        middleware = SubAgentMiddleware(
            subagents=[...],
            default_interrupt_on={
                "delete_file": True,
                "send_email": {"allowed_decisions": ["approve", "reject"]},
            },
            general_purpose_agent=True,
        )

        # Method 4: Subagent with custom interrupt_on (overrides default)
        middleware = SubAgentMiddleware(
            subagents=[
                SubAgentSpec(
                    name="careful_agent",
                    description="Agent that requires extra approval",
                    interrupt_on={"all_tools": True},  # Overrides default
                    ...
                )
            ],
            default_interrupt_on={"delete_file": True},
        )

        agent = ToolCallAgent(
            middleware=[middleware],
            ...
        )
        ```
    """

    system_prompt = TASK_SYSTEM_PROMPT

    def __init__(
        self,
        subagents: Optional[List[SubAgentType]] = None,
        default_middleware: Optional[List[AgentMiddleware]] = None,
        default_tools: Optional[List[BaseTool]] = None,
        default_interrupt_on: Optional[Dict[str, Union[bool, InterruptOnConfig]]] = None,
        max_depth: int = 3,
        general_purpose_agent: bool = True,
        task_description: Optional[str] = None,
    ):
        """Initialize subagent middleware.

        Args:
            subagents: List of subagent specifications (SubAgentSpec or CompiledSubAgent)
            default_middleware: Default middleware for all subagents
            default_tools: Default tools for general-purpose agent
            default_interrupt_on: Default HITL configuration for all subagents.
                This is used for the general-purpose agent and as a fallback
                for any SubAgentSpec that doesn't specify its own interrupt_on.
            max_depth: Maximum recursion depth for subagent delegation
            general_purpose_agent: Whether to include a general-purpose subagent (default: True)
            task_description: Custom description for the task tool.
                If None, uses default template.
                Supports {available_agents} placeholder for dynamic agent list.
        """
        super().__init__()
        self.subagent_specs = subagents or []
        self.default_middleware = default_middleware
        self.default_tools = default_tools
        self.default_interrupt_on = default_interrupt_on
        self.max_depth = max_depth
        self.general_purpose_agent = general_purpose_agent
        self.task_description = task_description

        # Manager will be initialized in before_agent
        self._manager: Optional[SubAgentManager] = None

        # Create temporary manager for tool collection
        self._temp_manager = SubAgentManager(
            parent_agent=None,
            subagent_specs=self.subagent_specs,
            default_middleware=default_middleware,
            default_tools=default_tools,
            default_interrupt_on=default_interrupt_on,
            max_depth=max_depth,
            general_purpose_agent=general_purpose_agent,
        )

        # Expose tools property for pipeline collection
        self.tools = [self._temp_manager.create_task_tool(task_description)]

        subagent_count = len(self.subagent_specs)
        if general_purpose_agent:
            subagent_count += 1
        logger.info(
            f"SubAgentMiddleware initialized with task tool for {subagent_count} subagents, "
            f"max_depth={max_depth}, has_interrupt_on={default_interrupt_on is not None}"
        )

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Initialize subagent manager with parent agent reference."""
        parent_agent = getattr(runtime, '_agent_instance', None)

        logger.debug(
            f"SubAgentMiddleware.before_agent called. parent_agent={parent_agent}, "
            f"has _manager={self._manager is not None}"
        )

        if parent_agent and not self._manager:
            self._temp_manager.parent = parent_agent
            self._manager = self._temp_manager
            logger.info(f"Initialized SubAgentManager with parent agent {parent_agent.name}")
        elif not parent_agent:
            logger.warning("before_agent called but parent_agent is None!")

        if "_subagent_depth" not in state:
            return {"_subagent_depth": 0}

        return None


# ============================================================================
# Integration Helper for BaseAgent
# ============================================================================

def add_subagent_support(
    agent: Any,
    subagents: List[SubAgentType],
    max_depth: int = 3,
    general_purpose_agent: bool = True,
    task_description: Optional[str] = None,
    default_interrupt_on: Optional[Dict[str, Union[bool, InterruptOnConfig]]] = None,
) -> Any:
    """Add subagent support to an existing agent.

    Args:
        agent: The agent instance
        subagents: List of subagent specifications (SubAgentSpec or CompiledSubAgent)
        max_depth: Maximum recursion depth for subagent delegation
        general_purpose_agent: Whether to include a general-purpose subagent
        task_description: Custom description for the task tool
        default_interrupt_on: Default HITL configuration for all subagents

    Returns:
        The agent with subagent support
    """
    manager = SubAgentManager(
        parent_agent=agent,
        subagent_specs=subagents,
        default_interrupt_on=default_interrupt_on,
        max_depth=max_depth,
        general_purpose_agent=general_purpose_agent,
    )

    task_tool = manager.create_task_tool(task_description)
    if hasattr(agent, 'available_tools'):
        agent.available_tools.add_tool(task_tool)

    agent._subagent_manager = manager

    subagent_count = len(subagents)
    if general_purpose_agent:
        subagent_count += 1
    logger.info(
        f"Added subagent support to agent '{agent.name}' with {subagent_count} subagents, "
        f"max_depth={max_depth}"
    )

    return agent


# ============================================================================
# Convenience Methods
# ============================================================================

def create_general_purpose_subagent(
    name: str = "general-purpose",
    description: str = DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    tools: Optional[List[BaseTool]] = None
) -> SubAgentSpec:
    """Create a general-purpose subagent specification."""
    return SubAgentSpec(
        name=name,
        description=description,
        system_prompt=DEFAULT_SUBAGENT_PROMPT,
        tools=tools or [],
        max_steps=10
    )


def create_compiled_subagent(
    name: str,
    description: str,
    graph: Any,
) -> CompiledSubAgent:
    """Create a CompiledSubAgent from a graph."""
    return CompiledSubAgent(
        name=name,
        description=description,
        runnable=graph,
    )
