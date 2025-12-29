"""
Middleware System for Deep Agents - Plan-Act-Reflect Support

This module provides a flexible middleware architecture that enables:
1. Plan-Act-Reflect cycles for deep reasoning
2. Model call interception and modification
3. Tool call wrapping and result transformation
4. Agent lifecycle hooks for state management
5. Declarative tool and state injection
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Callable, Dict, List, Optional, Union, TypeVar, Generic,
    AsyncIterator, Protocol
)
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import inspect
from copy import copy

from spoon_ai.schema import Message, Role, ToolCall, MessageContent
from spoon_ai.tools.base import BaseTool

logger = logging.getLogger(__name__)

# ============================================================================
# Agent Execution Phases - Explicit Plan-Act-Reflect Support
# ============================================================================

class AgentPhase(str, Enum):
    """Distinct phases in the agent execution cycle.

    This enables middleware to inject logic at specific thinking stages,
    supporting Plan-Act-Reflect and other deliberative patterns.
    """
    PLAN = "plan"           # Initial planning phase (before action loop)
    THINK = "think"         # LLM reasoning phase (selecting actions)
    ACT = "act"             # Tool execution phase
    OBSERVE = "observe"     # Observation/result processing phase
    REFLECT = "reflect"     # Reflection/evaluation phase
    FINISH = "finish"       # Completion phase


# ============================================================================
# Runtime Context - Shared state across middleware
# ============================================================================

@dataclass
class AgentRuntime:
    """Runtime context passed to middleware for state access.

    Provides safe access to agent state, configuration, and execution context
    without exposing the entire agent instance.
    """
    # Identifiers
    agent_name: str
    run_id: Optional[uuid.UUID] = None
    thread_id: Optional[str] = None

    # State access
    state: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # Execution context
    current_phase: AgentPhase = AgentPhase.THINK
    current_step: int = 0
    max_steps: int = 10

    # Message history (read-only reference)
    messages: List[Message] = field(default_factory=list)

    # Tool context
    tool_call_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Safely get state value."""
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set state value."""
        self.state[key] = value

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Bulk update state."""
        self.state.update(updates)

    def get_last_message(self, role: Optional[Role] = None) -> Optional[Message]:
        """Get last message, optionally filtered by role."""
        if not self.messages:
            return None

        if role is None:
            return self.messages[-1]

        for msg in reversed(self.messages):
            if msg.role == role.value:
                return msg
        return None


# ============================================================================
# Model Call Interception
# ============================================================================

@dataclass
class ModelRequest:
    """Request to LLM with full context."""
    # Prompt components
    system_prompt: Optional[str] = None
    messages: List[Message] = field(default_factory=list)

    # Tool context
    tools: List[Union[BaseTool, dict]] = field(default_factory=list)
    tool_choice: str = "auto"

    # Model configuration
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096

    # Runtime context
    runtime: Optional[AgentRuntime] = None

    # Phase context (for Plan-Act-Reflect awareness)
    phase: AgentPhase = AgentPhase.THINK

    # Additional provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def override(self, **kwargs) -> "ModelRequest":
        """Create a modified copy with updated fields.

        Example:
            new_req = request.override(
                system_prompt="Updated prompt",
                temperature=0.7
            )
        """
        new_req = copy(self)
        for key, value in kwargs.items():
            if hasattr(new_req, key):
                setattr(new_req, key, value)
            else:
                new_req.extra_params[key] = value
        return new_req

    def append_to_system_prompt(self, text: str) -> "ModelRequest":
        """Convenience method to append text to system prompt."""
        if self.system_prompt:
            new_prompt = f"{self.system_prompt}\n\n{text}"
        else:
            new_prompt = text
        return self.override(system_prompt=new_prompt)

    def add_tool(self, tool: Union[BaseTool, dict]) -> "ModelRequest":
        """Add a tool to the request."""
        new_tools = self.tools.copy()
        new_tools.append(tool)
        return self.override(tools=new_tools)


@dataclass
class ModelResponse:
    """Response from LLM."""
    # Content
    content: str

    # Tool calls
    tool_calls: List[ToolCall] = field(default_factory=list)

    # Metadata
    finish_reason: Optional[str] = None
    model: Optional[str] = None

    # Usage statistics
    usage: Optional[Dict[str, int]] = None

    # Provider-specific data
    raw_response: Optional[Any] = None


# ============================================================================
# Tool Call Interception
# ============================================================================

@dataclass
class ToolCallRequest:
    """Request to execute a tool."""
    # Tool identification
    tool_name: str
    arguments: Dict[str, Any]
    tool_call_id: str

    # Runtime context
    runtime: Optional[AgentRuntime] = None

    # Original tool call object
    tool_call: Optional[ToolCall] = None


@dataclass
class ToolCallResult:
    """Result of tool execution."""
    # Primary result
    output: str

    # Error information
    error: Optional[str] = None

    # State updates (for backends like StateBackend in deepagents)
    state_updates: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if tool call was successful."""
        return self.error is None

    @staticmethod
    def from_string(output: str) -> "ToolCallResult":
        """Create result from simple string output."""
        return ToolCallResult(output=output)

    @staticmethod
    def from_error(error: str) -> "ToolCallResult":
        """Create error result."""
        return ToolCallResult(output="", error=error)


# ============================================================================
# Phase Hook Protocol - For Plan-Act-Reflect
# ============================================================================

class PhaseHook(Protocol):
    """Protocol for phase-specific hooks.

    Middleware can implement phase hooks to inject logic at specific
    points in the agent's execution cycle (Plan, Act, Reflect, etc.)
    """

    def __call__(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute phase hook.

        Args:
            runtime: Agent runtime context
            phase_data: Phase-specific data (e.g., plan, observations, reflections)

        Returns:
            Optional state updates to merge into agent state
        """
        ...


# ============================================================================
# Agent Middleware Base Class
# ============================================================================

class AgentMiddleware(ABC):
    """Base class for agent middleware with Plan-Act-Reflect support.

    Middleware can:
    1. Inject tools and extend agent capabilities
    2. Modify system prompts
    3. Intercept and transform model calls
    4. Wrap tool executions
    5. Hook into agent lifecycle events
    6. Implement Plan-Act-Reflect phases

    Example:
        class PlanningMiddleware(AgentMiddleware):
            tools = [create_plan_tool(), update_plan_tool()]
            system_prompt = "You create and follow detailed plans."

            def on_plan_phase(self, runtime, phase_data):
                # Generate initial plan before action loop
                return {"plan": self._generate_plan(runtime)}

            def on_reflect_phase(self, runtime, phase_data):
                # Evaluate progress and update plan
                plan_success = self._evaluate_plan(runtime)
                if not plan_success:
                    return {"plan": self._revise_plan(runtime)}
                return None
    """

    # ========================================================================
    # Declarative Configuration
    # ========================================================================

    # Tools to inject into agent
    tools: List[BaseTool] = []

    # System prompt to append
    system_prompt: Optional[str] = None

    # State schema extension (for typed state)
    state_schema: Optional[type] = None

    # This prevents sharing hooks across middleware instances

    def __init__(self):
        """Initialize middleware with instance-specific phase hooks registry."""
        super().__init__()
        self._phase_hooks: Dict[AgentPhase, PhaseHook] = {}

    # ========================================================================
    # Model Call Hooks
    # ========================================================================

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Intercept LLM call (synchronous).

        Override to modify requests/responses or inject side effects.

        Args:
            request: Model request with full context
            handler: Next handler in the middleware chain

        Returns:
            Model response (potentially modified)

        Example:
            def wrap_model_call(self, request, handler):
                # Log request
                logger.info(f"Calling model with {len(request.messages)} messages")

                # Modify request
                if request.phase == AgentPhase.PLAN:
                    request = request.append_to_system_prompt(
                        "Create a detailed step-by-step plan."
                    )

                # Execute
                response = handler(request)

                # Post-process response
                if request.phase == AgentPhase.REFLECT:
                    self._store_reflection(response.content)

                return response
        """
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Intercept LLM call (asynchronous).

        IMPORTANT: In async pipelines, override this method instead of wrap_model_call.
        If you override wrap_model_call, the handler will be async and return a coroutine.

        Default implementation: pass through to handler.
        """
        # Check if user overrode wrap_model_call (sync version)
        has_custom_sync = (
            type(self).wrap_model_call is not AgentMiddleware.wrap_model_call
        )

        # Check if user overrode awrap_model_call (this method)
        has_custom_async = (
            type(self).awrap_model_call is not AgentMiddleware.awrap_model_call
        )

        if has_custom_sync and not has_custom_async:
            # User only overrode sync version - this won't work in async pipeline!
            # Try to call it, but handler will return a coroutine
            logger.warning(
                f"{type(self).__name__} only overrides wrap_model_call (sync). "
                f"In async pipelines, override awrap_model_call instead. "
                f"Handler will be async!"
            )

            result = self.wrap_model_call(request, handler)

            # If it returned a coroutine, await it
            if inspect.iscoroutine(result):
                return await result
            return result

        # Default: just call handler
        return await handler(request)

    # ========================================================================
    # Tool Call Hooks
    # ========================================================================

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult]
    ) -> ToolCallResult:
        """Intercept tool execution (synchronous).

        Override to:
        - Log tool usage
        - Modify tool arguments
        - Transform tool results
        - Implement approval flows (HITL)
        - Handle large results (save to filesystem)

        Args:
            request: Tool call request
            handler: Next handler in the middleware chain

        Returns:
            Tool call result (potentially modified)

        Example:
            def wrap_tool_call(self, request, handler):
                # Approval for dangerous tools
                if request.tool_name in self.require_approval:
                    if not self._get_approval(request):
                        return ToolCallResult.from_error("User rejected tool call")

                # Execute
                result = handler(request)

                # Handle large results
                if len(result.output) > 20000:
                    path = self._save_to_file(result.output)
                    return ToolCallResult(
                        output=f"Result saved to {path} (too large to display)",
                        metadata={"saved_path": path}
                    )

                return result
        """
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult]
    ) -> ToolCallResult:
        """Intercept tool execution (asynchronous).

        Default implementation delegates to sync version and handles async handlers.
        """
        result = self.wrap_tool_call(request, handler)

        # it returns a coroutine object. We need to await it.
        if inspect.iscoroutine(result):
            return await result

        return result

    # ========================================================================
    # Agent Lifecycle Hooks
    # ========================================================================

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Called before agent starts processing.

        Use for:
        - State initialization
        - Message preprocessing
        - Context setup

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            Optional state updates to merge

        Example:
            def before_agent(self, state, runtime):
                # Initialize planning state
                if "plan" not in state:
                    return {"plan": None, "plan_steps": []}
                return None
        """
        return None

    def after_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Called after agent finishes processing.

        Use for:
        - State cleanup
        - Result post-processing
        - Logging/metrics

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            Optional state updates to merge
        """
        return None

    # ========================================================================
    # Phase-Specific Hooks (Plan-Act-Reflect)
    # ========================================================================

    def on_plan_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Called during PLAN phase (before action loop).

        Override to implement planning logic:
        - Generate initial plan from user request
        - Break down complex tasks
        - Set up execution strategy

        Args:
            runtime: Runtime context with user message
            phase_data: Phase-specific data (may contain previous plan)

        Returns:
            State updates (e.g., {"plan": plan_object, "steps": [...]})

        Example:
            def on_plan_phase(self, runtime, phase_data):
                user_request = runtime.get_last_message(Role.USER)
                plan = self._create_plan(user_request.content)
                return {
                    "plan": plan,
                    "plan_steps": plan.steps,
                    "plan_created_at": datetime.now().isoformat()
                }
        """
        return None

    def on_think_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Called during THINK phase (before LLM reasoning).

        Override to:
        - Inject planning context into prompts
        - Filter available tools based on plan
        - Add think-aloud prompts

        Args:
            runtime: Runtime context
            phase_data: Phase data (may contain current plan step)

        Returns:
            State updates
        """
        return None

    def on_act_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Called during ACT phase (before tool execution).

        Override to:
        - Validate actions against plan
        - Log action decisions
        - Update plan progress

        Args:
            runtime: Runtime context
            phase_data: Phase data (contains selected tool calls)

        Returns:
            State updates
        """
        return None

    def on_observe_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Called during OBSERVE phase (after tool execution).

        Override to:
        - Process tool results
        - Update world model
        - Extract key observations

        Args:
            runtime: Runtime context
            phase_data: Phase data (contains tool results)

        Returns:
            State updates
        """
        return None

    def on_reflect_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Called during REFLECT phase (after N steps or on completion).

        Override to:
        - Evaluate progress against plan
        - Decide if plan needs revision
        - Identify stuck states
        - Trigger replanning

        Args:
            runtime: Runtime context
            phase_data: Phase data (progress, observations, current plan)

        Returns:
            State updates (e.g., {"plan": revised_plan, "needs_replan": False})

        Example:
            def on_reflect_phase(self, runtime, phase_data):
                plan = runtime.get_state("plan")
                progress = phase_data.get("progress", {})

                if not self._is_making_progress(progress):
                    new_plan = self._revise_plan(plan, progress)
                    return {
                        "plan": new_plan,
                        "plan_revision": runtime.get_state("plan_revision", 0) + 1
                    }

                return None
        """
        return None

    def on_finish_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Called during FINISH phase (before agent returns).

        Override to:
        - Validate completion
        - Generate final report
        - Clean up temporary state

        Args:
            runtime: Runtime context
            phase_data: Phase data (final result, statistics)

        Returns:
            State updates
        """
        return None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def register_phase_hook(self, phase: AgentPhase, hook: PhaseHook) -> None:
        """Register a custom phase hook dynamically.

        Args:
            phase: Target phase
            hook: Callable hook function
        """
        if not hasattr(self, '_phase_hooks'):
            self._phase_hooks = {}
        self._phase_hooks[phase] = hook

    def get_phase_hook(self, phase: AgentPhase) -> Optional[PhaseHook]:
        """Get registered hook for a phase.

        Returns the method (on_plan_phase, etc.) or custom registered hook.
        """
        # Check for method
        method_name = f"on_{phase.value}_phase"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            # Only return if it's been overridden (not the base implementation)
            base_method = getattr(AgentMiddleware, method_name)
            try:
                method_func = method.__func__  # bound method
            except AttributeError:
                method_func = method  # function assigned on instance
            try:
                base_func = base_method.__func__
            except AttributeError:
                base_func = base_method
            if method_func is not base_func:
                return method

        # Check for registered hook
        return getattr(self, '_phase_hooks', {}).get(phase)


# ============================================================================
# Middleware Pipeline - Composes multiple middleware
# ============================================================================

class MiddlewarePipeline:
    """Composes multiple middleware into an execution pipeline.

    Handles:
    - Onion wrapping of model/tool calls
    - Sequential execution of lifecycle hooks
    - Phase hook orchestration
    - Tool aggregation
    - System prompt composition
    """

    def __init__(self, middleware_list: List[AgentMiddleware]):
        """Initialize pipeline with middleware list.

        Args:
            middleware_list: List of middleware in application order
        """
        self.middleware = middleware_list
        logger.info(f"Initialized middleware pipeline with {len(middleware_list)} middleware")

    # ========================================================================
    # Model Call Pipeline
    # ========================================================================

    def wrap_model_call(
        self,
        request: ModelRequest,
        base_handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Build onion of middleware around base model handler.

        Execution order: last middleware wraps first (reverse order).
        """
        handler = base_handler

        # Wrap in reverse order
        for mw in reversed(self.middleware):
            current_handler = handler
            # Capture mw and current_handler in closure
            def make_wrapper(middleware, inner_handler):
                return lambda req: middleware.wrap_model_call(req, inner_handler)
            handler = make_wrapper(mw, current_handler)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        base_handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Async version of model call pipeline."""
        handler = base_handler

        # It's a closure factory, not a coroutine
        for mw in reversed(self.middleware):
            current_handler = handler
            def make_async_wrapper(middleware, inner_handler):
                async def wrapper(req):
                    return await middleware.awrap_model_call(req, inner_handler)
                return wrapper
            handler = make_async_wrapper(mw, current_handler)  # No await!

        return await handler(request)

    # ========================================================================
    # Tool Call Pipeline
    # ========================================================================

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        base_handler: Callable[[ToolCallRequest], ToolCallResult]
    ) -> ToolCallResult:
        """Build onion of middleware around base tool handler."""
        handler = base_handler

        for mw in reversed(self.middleware):
            current_handler = handler
            def make_wrapper(middleware, inner_handler):
                return lambda req: middleware.wrap_tool_call(req, inner_handler)
            handler = make_wrapper(mw, current_handler)

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        base_handler: Callable[[ToolCallRequest], ToolCallResult]
    ) -> ToolCallResult:
        """Async version of tool call pipeline."""
        handler = base_handler

        
        # It's a closure factory, not a coroutine
        for mw in reversed(self.middleware):
            current_handler = handler
            def make_async_wrapper(middleware, inner_handler):
                async def wrapper(req):
                    return await middleware.awrap_tool_call(req, inner_handler)
                return wrapper
            handler = make_async_wrapper(mw, current_handler)  # No await!

        return await handler(request)

    # ========================================================================
    # Lifecycle Hooks
    # ========================================================================

    def execute_before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Dict[str, Any]:
        """Execute before_agent hooks in middleware order.

        Returns merged state updates from all middleware.
        """
        logger.info(f"🔄 Executing before_agent hooks for {len(self.middleware)} middleware")
        merged_updates = {}

        for mw in self.middleware:
            try:
                logger.info(f"   → Calling {mw.__class__.__name__}.before_agent")
                updates = mw.before_agent(state, runtime)
                if updates:
                    merged_updates.update(updates)
                    state.update(updates)  # Apply incrementally for next middleware
                    logger.info(f"   ✓ {mw.__class__.__name__} updated state: {list(updates.keys())}")
            except Exception as e:
                logger.error(f"Error in {mw.__class__.__name__}.before_agent: {e}")

        return merged_updates

    def execute_after_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Dict[str, Any]:
        """Execute after_agent hooks in reverse middleware order."""
        logger.info(f"🔄 Executing after_agent hooks for {len(self.middleware)} middleware (reverse order)")
        merged_updates = {}

        for mw in reversed(self.middleware):
            try:
                logger.info(f"   → Calling {mw.__class__.__name__}.after_agent")
                updates = mw.after_agent(state, runtime)
                if updates:
                    merged_updates.update(updates)
                    state.update(updates)
                    logger.info(f"   ✓ {mw.__class__.__name__} updated state: {list(updates.keys())}")
            except Exception as e:
                logger.error(f"Error in {mw.__class__.__name__}.after_agent: {e}")

        return merged_updates

    # ========================================================================
    # Phase Hook Execution
    # ========================================================================

    def execute_phase_hook(
        self,
        phase: AgentPhase,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute phase hooks from all middleware.

        Args:
            phase: Current execution phase
            runtime: Runtime context
            phase_data: Phase-specific data

        Returns:
            Merged state updates from all middleware
        """
        merged_updates = {}

        logger.debug(f"Executing {phase.value} phase hooks for {len(self.middleware)} middleware")

        for mw in self.middleware:
            try:
                hook = mw.get_phase_hook(phase)
                if hook:
                    updates = hook(runtime, phase_data)
                    if updates:
                        merged_updates.update(updates)
                        runtime.update_state(updates)
                        logger.debug(f"{mw.__class__.__name__} updated state in {phase.value} phase")
            except Exception as e:
                logger.error(f"Error in {mw.__class__.__name__}.on_{phase.value}_phase: {e}")

        return merged_updates

    # ========================================================================
    # Tool Aggregation
    # ========================================================================

    def collect_tools(self) -> List[BaseTool]:
        """Aggregate tools from all middleware."""
        tools = []
        for mw in self.middleware:
            if hasattr(mw, 'tools') and mw.tools:
                tools.extend(mw.tools)
                logger.debug(f"{mw.__class__.__name__} contributed {len(mw.tools)} tools")

        logger.info(f"Collected {len(tools)} total tools from middleware")
        return tools

    # ========================================================================
    # System Prompt Composition
    # ========================================================================

    def build_system_prompt(self, base_prompt: Optional[str] = None) -> str:
        """Build composed system prompt from base + middleware prompts.
        Args:
            base_prompt: Agent's base system prompt

        Returns:
            Composed prompt with middleware additions
        """
        parts = []

        # Middleware prompts should still be applied
        if base_prompt:
            parts.append(base_prompt)

        for mw in self.middleware:
            if hasattr(mw, 'system_prompt') and mw.system_prompt:
                parts.append(mw.system_prompt)
                logger.debug(f"{mw.__class__.__name__} added system prompt section")
        if not parts:
            return ""

        composed = "\n\n".join(parts)
        logger.info(f"Composed system prompt: {len(composed)} chars from {len(parts)} sections")
        return composed


# ============================================================================
# Convenience Functions
# ============================================================================

def create_middleware_pipeline(
    middleware: List[Union[AgentMiddleware, type]]
) -> MiddlewarePipeline:
    """Create middleware pipeline from middleware classes or instances.

    Args:
        middleware: List of AgentMiddleware instances or classes

    Returns:
        Configured middleware pipeline

    Example:
        pipeline = create_middleware_pipeline([
            PlanningMiddleware(),
            ObservabilityMiddleware(),
            FilesystemMiddleware
        ])
    """
    instances = []
    for mw in middleware:
        if isinstance(mw, type):
            # Instantiate class
            instances.append(mw())
        else:
            # Already an instance
            instances.append(mw)

    return MiddlewarePipeline(instances)
