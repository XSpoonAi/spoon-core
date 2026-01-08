"""
Human-in-the-Loop (HITL) System

Provides approval workflows for critical tool executions.
Compatible with LangChain DeepAgents HumanInTheLoopMiddleware interface.

Features:
- Tool-level approval configuration
- Multiple approval strategies (approve, edit, reject)
- Batch interrupt/resume support for parallel tool calls
- Command(resume=...) pattern for resuming
- Checkpointer integration for state persistence
- Return value mode (__interrupt__) instead of exceptions

Usage:
    from spoon_ai.tools.hitl import HumanInTheLoopMiddleware
    from spoon_ai.graph import Command, InMemoryCheckpointer

    # Create checkpointer (REQUIRED for HITL)
    checkpointer = InMemoryCheckpointer()

    # Configure middleware
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "delete_file": True,
            "send_email": {"allowed_decisions": ["approve", "reject"]}
        },
        checkpointer=checkpointer
    )

    # Create agent
    agent = ToolCallAgent(
        tools=[dangerous_tool],
        middleware=[middleware],
        thread_id="my-thread"
    )

    # Run agent
    result = await agent.run(task)

    # Check for interrupt
    if result.get("__interrupt__"):
        interrupts = result["__interrupt__"][0]["value"]
        action_requests = interrupts["action_requests"]
        review_configs = interrupts["review_configs"]

        # Resume with decisions
        result = await agent.run(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config={"configurable": {"thread_id": "my-thread"}}
        )
"""

import json
import logging
import uuid
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Literal,
    Callable,
    Union,
    TypedDict,
    Protocol,
)
from dataclasses import dataclass, field
from enum import Enum

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Checkpointer Protocol (compatible with graph.checkpointer)
# ============================================================================

class CheckpointerProtocol(Protocol):
    """Protocol for checkpointer compatibility."""

    def save_checkpoint(self, thread_id: str, snapshot: Any) -> None:
        """Save a checkpoint."""
        ...

    def get_checkpoint(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Any]:
        """Get a checkpoint."""
        ...


# ============================================================================
# Type Definitions - Compatible with LangChain
# ============================================================================

class InterruptOnConfig(TypedDict, total=False):
    """Configuration for tool interruption.

    Compatible with LangChain DeepAgents InterruptOnConfig.

    Attributes:
        allowed_decisions: List of allowed approval decisions.
            Defaults to ["approve", "edit", "reject"].

    Example:
        config: InterruptOnConfig = {
            "allowed_decisions": ["approve", "reject"],
        }
    """

    allowed_decisions: List[Literal["approve", "edit", "reject"]]


# ============================================================================
# Approval Decision Types
# ============================================================================

class ApprovalDecision(str, Enum):
    """Possible approval decisions."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


@dataclass
class DecisionInput:
    """Input for a single approval decision.

    Compatible with LangChain format:
    - {"type": "approve"}
    - {"type": "reject"}
    - {"type": "edit", "edited_action": {"name": "tool_name", "args": {...}}}
    """

    type: Literal["approve", "edit", "reject"]
    edited_action: Optional[Dict[str, Any]] = None  # For edit decision (LangChain format)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionInput":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "approve"),
            edited_action=data.get("edited_action"),
        )


# ============================================================================
# Action Request Types (LangChain Compatible)
# ============================================================================

@dataclass
class ActionRequest:
    """A pending action request.

    Compatible with LangChain's action_request format.
    """

    name: str  # Tool name
    args: Dict[str, Any]  # Tool arguments
    id: str  # Tool call ID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "args": self.args,
            "id": self.id,
        }


@dataclass
class ReviewConfig:
    """Review configuration for a pending action.

    Compatible with LangChain's review_config format.
    """

    action_name: str
    action_id: str
    allowed_decisions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_name": self.action_name,
            "action_id": self.action_id,
            "allowed_decisions": self.allowed_decisions,
        }


@dataclass
class InterruptValue:
    """Value returned in __interrupt__ for batch interrupts.

    Compatible with LangChain's interrupt value format.
    Contains both action_requests and review_configs.
    """

    action_requests: List[ActionRequest]
    review_configs: List[ReviewConfig]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_requests": [r.to_dict() for r in self.action_requests],
            "review_configs": [c.to_dict() for c in self.review_configs],
        }


@dataclass
class InterruptInfo:
    """Wrapper for interrupt information.

    Compatible with LangChain's __interrupt__[0] format.
    """

    value: InterruptValue
    interrupt_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value.to_dict(),
            "interrupt_id": self.interrupt_id,
        }


# ============================================================================
# Parsed Config Helper
# ============================================================================

@dataclass
class ParsedInterruptConfig:
    """Parsed and normalized interrupt configuration."""

    allowed_decisions: List[ApprovalDecision]

    @classmethod
    def from_config(cls, config: Union[bool, InterruptOnConfig]) -> Optional["ParsedInterruptConfig"]:
        """Parse configuration from various formats."""
        if isinstance(config, bool):
            if config:
                return cls(
                    allowed_decisions=[
                        ApprovalDecision.APPROVE,
                        ApprovalDecision.EDIT,
                        ApprovalDecision.REJECT,
                    ]
                )
            return None

        if isinstance(config, dict):
            allowed = config.get("allowed_decisions", ["approve", "edit", "reject"])
            return cls(
                allowed_decisions=[ApprovalDecision(d) for d in allowed],
            )

        return None


# ============================================================================
# HITL State for Checkpointing
# ============================================================================

@dataclass
class HITLState:
    """State for HITL that can be checkpointed."""

    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    pending_configs: List[Dict[str, Any]] = field(default_factory=list)
    interrupt_id: Optional[str] = None
    is_interrupted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for checkpointing."""
        return {
            "pending_actions": self.pending_actions,
            "pending_configs": self.pending_configs,
            "interrupt_id": self.interrupt_id,
            "is_interrupted": self.is_interrupted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HITLState":
        """Create from dictionary."""
        return cls(
            pending_actions=data.get("pending_actions", []),
            pending_configs=data.get("pending_configs", []),
            interrupt_id=data.get("interrupt_id"),
            is_interrupted=data.get("is_interrupted", False),
        )


# ============================================================================
# HITL Manager
# ============================================================================

class HITLManager:
    """Manages human-in-the-loop approval workflows.

    Supports batch interrupts for parallel tool calls.
    Uses return value mode (__interrupt__) instead of exceptions.
    """

    def __init__(
        self,
        interrupt_on: Dict[str, Union[bool, InterruptOnConfig]],
        checkpointer: Optional[CheckpointerProtocol] = None,
    ):
        """Initialize HITL manager.

        Args:
            interrupt_on: Tool interruption configuration.
                - Dict[str, bool]: Simple approval (True = require approval)
                - Dict[str, InterruptOnConfig]: Detailed configuration with
                  allowed_decisions
            checkpointer: Optional checkpointer for state persistence.
                Required for proper interrupt/resume workflow.
        """
        self.interrupt_on = interrupt_on
        self.checkpointer = checkpointer

        # Parse configurations
        self.tool_configs: Dict[str, ParsedInterruptConfig] = {}
        for tool_name, config in interrupt_on.items():
            parsed = ParsedInterruptConfig.from_config(config)
            if parsed:
                self.tool_configs[tool_name] = parsed

        # Current HITL state
        self._state = HITLState()

        logger.info(f"HITL Manager initialized with {len(self.tool_configs)} tool configurations")

    def should_interrupt(self, tool_name: str) -> bool:
        """Check if tool requires approval."""
        return tool_name in self.tool_configs

    def get_config(self, tool_name: str) -> Optional[ParsedInterruptConfig]:
        """Get parsed configuration for a tool."""
        return self.tool_configs.get(tool_name)

    def add_pending_action(self, tool_call: Any) -> None:
        """Add a pending action for batch interrupt.

        Args:
            tool_call: The ToolCall object with function.name, function.arguments, id
        """
        tool_name = tool_call.function.name
        tool_id = tool_call.id

        # Parse arguments from JSON string
        try:
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}

        config = self.tool_configs.get(tool_name)
        if not config:
            return

        # Create action request
        action = ActionRequest(
            name=tool_name,
            args=args,
            id=tool_id,
        )
        self._state.pending_actions.append(action.to_dict())

        # Create review config
        review = ReviewConfig(
            action_name=tool_name,
            action_id=tool_id,
            allowed_decisions=[d.value for d in config.allowed_decisions],
        )
        self._state.pending_configs.append(review.to_dict())

        logger.debug(f"Added pending action: {tool_name} (ID: {tool_id})")

    def has_pending_actions(self) -> bool:
        """Check if there are pending actions."""
        return len(self._state.pending_actions) > 0

    def create_interrupt(self) -> Dict[str, Any]:
        """Create an interrupt with all pending actions.

        Returns:
            Dictionary in __interrupt__ format compatible with LangChain.
        """
        action_requests = [
            ActionRequest(**a) for a in self._state.pending_actions
        ]
        review_configs = [
            ReviewConfig(**c) for c in self._state.pending_configs
        ]

        interrupt_value = InterruptValue(
            action_requests=action_requests,
            review_configs=review_configs,
        )

        interrupt_info = InterruptInfo(value=interrupt_value)
        self._state.interrupt_id = interrupt_info.interrupt_id
        self._state.is_interrupted = True

        logger.info(f"Created HITL interrupt with {len(action_requests)} actions")

        # Return in LangChain format: __interrupt__ is a list
        return {
            "__interrupt__": [interrupt_info.to_dict()]
        }

    def clear_pending(self) -> None:
        """Clear all pending actions."""
        self._state.pending_actions.clear()
        self._state.pending_configs.clear()

    def reset_state(self) -> None:
        """Reset HITL state completely."""
        self._state = HITLState()

    def save_state(self, thread_id: str) -> None:
        """Save current state to checkpointer."""
        if self.checkpointer is None:
            logger.warning("No checkpointer configured, HITL state not saved")
            return

        from spoon_ai.graph.types import StateSnapshot
        from datetime import datetime

        snapshot = StateSnapshot(
            values={"hitl_state": self._state.to_dict()},
            next=(),
            config={"configurable": {"thread_id": thread_id}},
            metadata={"type": "hitl_interrupt"},
            created_at=datetime.now(),
        )

        self.checkpointer.save_checkpoint(thread_id, snapshot)
        logger.debug(f"Saved HITL state for thread {thread_id}")

    def load_state(self, thread_id: str) -> bool:
        """Load state from checkpointer.

        Returns:
            True if state was loaded, False otherwise.
        """
        if self.checkpointer is None:
            return False

        snapshot = self.checkpointer.get_checkpoint(thread_id)
        if snapshot is None:
            return False

        hitl_data = snapshot.values.get("hitl_state")
        if hitl_data:
            self._state = HITLState.from_dict(hitl_data)
            logger.debug(f"Loaded HITL state for thread {thread_id}")
            return True

        return False

    def apply_decisions(
        self,
        decisions: List[DecisionInput],
        tool_calls: List[Any],
    ) -> List[Any]:
        """Apply decisions to tool calls.

        Args:
            decisions: List of decisions from Command(resume=...)
            tool_calls: Original tool calls

        Returns:
            Modified tool calls with edits applied, rejected calls removed
        """
        from spoon_ai.schema import ToolCall as ToolCallSchema, Function

        result = []

        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.function.name
            if i < len(decisions):
                decision = decisions[i]
                if decision.type == "reject":
                    logger.info(f"Tool {tool_name} rejected")
                    continue
                elif decision.type == "edit":
                    # Apply edited arguments - LangChain format
                    if decision.edited_action and "args" in decision.edited_action:
                        new_args = json.dumps(decision.edited_action["args"])
                    else:
                        new_args = tool_call.function.arguments

                    new_function = Function(name=tool_name, arguments=new_args)
                    modified = ToolCallSchema(id=tool_call.id, type=tool_call.type, function=new_function)
                    result.append(modified)
                    logger.info(f"Tool {tool_name} approved with edits")
                else:  # approve
                    result.append(tool_call)
                    logger.info(f"Tool {tool_name} approved")
            else:
                result.append(tool_call)

        # Clear interrupted state after applying decisions
        self._state.is_interrupted = False
        self.clear_pending()

        return result


# ============================================================================
# Human-in-the-Loop Middleware (LangChain Compatible)
# ============================================================================

class HumanInTheLoopMiddleware(AgentMiddleware):
    """Middleware that implements Human-in-the-Loop approval workflows.

    Compatible with LangChain DeepAgents HumanInTheLoopMiddleware.

    This middleware intercepts tool calls that require approval and returns
    an __interrupt__ with action_requests and review_configs for batch approval.

    Features:
    - Per-tool approval configuration
    - Batch interrupt/resume for parallel tool calls
    - Command(resume=...) pattern for resuming
    - Checkpointer integration for state persistence
    - Return value mode (__interrupt__) instead of exceptions

    Usage:
        from spoon_ai.graph import InMemoryCheckpointer

        # Checkpointer is REQUIRED for human-in-the-loop
        checkpointer = InMemoryCheckpointer()

        middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_file": True,  # Default: approve, edit, reject
                "read_file": False,   # No interrupts needed
                "send_email": {"allowed_decisions": ["approve", "reject"]},
            },
            checkpointer=checkpointer
        )

        # Resume from interrupt
        result = agent.run(
            Command(resume={"decisions": [
                {"type": "approve"},
                {"type": "edit", "edited_action": {"name": "send_email", "args": {...}}},
            ]}),
            config={"configurable": {"thread_id": "my-thread"}}
        )

    Interrupt Format (returned in result):
        {
            "__interrupt__": [
                {
                    "value": {
                        "action_requests": [
                            {"name": "delete_file", "args": {"path": "/tmp/test"}, "id": "..."},
                        ],
                        "review_configs": [
                            {
                                "action_name": "delete_file",
                                "action_id": "...",
                                "allowed_decisions": ["approve", "edit", "reject"],
                            },
                        ],
                    },
                    "interrupt_id": "...",
                }
            ]
        }
    """

    system_prompt = """# Human-in-the-Loop Approval

Some tools require human approval before execution. When you attempt to use
these tools, execution will pause and return an interrupt with pending actions.

Approval workflow:
1. Tool calls are collected
2. Interrupt is returned with action_requests and review_configs
3. User provides decisions (approve/edit/reject) for each action
4. Execution resumes with Command(resume={"decisions": [...]})

Be prepared for tool calls to be rejected or modified.
"""

    def __init__(
        self,
        interrupt_on: Dict[str, Union[bool, InterruptOnConfig]],
        checkpointer: Optional[CheckpointerProtocol] = None,
    ):
        """Initialize HITL middleware.

        Args:
            interrupt_on: Tool interruption configuration.
                Keys are tool names, values can be:
                - True: Require approval with default allowed_decisions
                - False: No approval needed (tool is skipped)
                - InterruptOnConfig dict with:
                    - allowed_decisions: List of ["approve", "edit", "reject"]
            checkpointer: Optional checkpointer for state persistence.
                Required for proper interrupt/resume workflow.
        """
        super().__init__()
        self.interrupt_on = interrupt_on
        self.checkpointer = checkpointer
        self.manager = HITLManager(interrupt_on, checkpointer)

        # Track pending resume data from Command
        self._resume_data: Optional[Dict[str, Any]] = None
        self._current_thread_id: Optional[str] = None

        logger.info(f"HumanInTheLoopMiddleware initialized for {len(self.manager.tool_configs)} tools")

    def set_resume_data(self, resume: Dict[str, Any], thread_id: Optional[str] = None) -> None:
        """Set resume data from Command(resume=...).

        Called by the agent when resuming from interrupt.

        Args:
            resume: Resume data with decisions
            thread_id: Thread ID for loading checkpoint state
        """
        self._resume_data = resume

        # Load state from checkpointer if available
        if thread_id and self.checkpointer:
            self.manager.load_state(thread_id)

        decisions = resume.get("decisions", [])
        logger.info(f"Resume data set with {len(decisions)} decisions")

    def before_agent(
        self,
        state: Dict[str, Any],  # noqa: ARG002
        runtime: AgentRuntime,
    ) -> Optional[Dict[str, Any]]:
        """Initialize state for HITL tracking."""
        # Get thread_id from runtime if available
        if runtime and hasattr(runtime, '_agent_instance'):
            agent = runtime._agent_instance
            if hasattr(agent, 'thread_id'):
                self._current_thread_id = agent.thread_id

        # Clear pending actions if not resuming
        if self._resume_data is None:
            self.manager.clear_pending()

        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Intercept model response to collect tool calls requiring approval.

        This processes the model response and identifies tool calls that
        need approval, then returns an interrupt if any are found.
        """
        # Call the model
        response = await handler(request)

        # Check if there are tool calls requiring approval
        if not response.tool_calls:
            return response

        # Check if we're resuming from interrupt
        if self._resume_data:
            # Parse decisions
            decisions_data = self._resume_data.get("decisions", [])
            decisions = [DecisionInput.from_dict(d) for d in decisions_data]
            self._resume_data = None

            # Apply decisions to tool calls
            approved_calls = self.manager.apply_decisions(decisions, response.tool_calls)

            # Update response with approved calls
            response.tool_calls = approved_calls

            return response

        # Collect tool calls requiring approval
        tools_requiring_approval = []
        tools_not_requiring_approval = []

        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            if self.manager.should_interrupt(tool_name):
                tools_requiring_approval.append(tool_call)
                self.manager.add_pending_action(tool_call)
            else:
                tools_not_requiring_approval.append(tool_call)

        # If there are tools requiring approval, create interrupt
        if self.manager.has_pending_actions():
            interrupt_data = self.manager.create_interrupt()

            # Save state to checkpointer
            if self._current_thread_id:
                self.manager.save_state(self._current_thread_id)

            # Store interrupt data in response for agent to handle
            response.interrupt = interrupt_data
            # Clear tool calls that need approval - they'll be re-executed after resume
            response.tool_calls = tools_not_requiring_approval

            logger.info(f"HITL interrupt created, {len(tools_requiring_approval)} tools pending approval")

        return response

    def get_interrupt_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get interrupt configuration for a tool.

        Returns:
            Dict with allowed_decisions, or None if tool doesn't require approval
        """
        config = self.manager.get_config(tool_name)
        if not config:
            return None

        return {
            "allowed_decisions": [d.value for d in config.allowed_decisions],
        }


# ============================================================================
# Legacy Alias for Backward Compatibility
# ============================================================================

HITLMiddleware = HumanInTheLoopMiddleware


# ============================================================================
# Convenience Functions
# ============================================================================

def create_hitl_middleware(
    *tool_names: str,
    allowed_decisions: Optional[List[str]] = None,
    checkpointer: Optional[CheckpointerProtocol] = None,
) -> HumanInTheLoopMiddleware:
    """Create HITL middleware for specified tools.

    Args:
        *tool_names: Names of tools that require approval
        allowed_decisions: Allowed decisions for all tools
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Configured HumanInTheLoopMiddleware

    Example:
        from spoon_ai.graph import InMemoryCheckpointer

        checkpointer = InMemoryCheckpointer()
        middleware = create_hitl_middleware(
            "delete_file",
            "send_email",
            "shutdown_server",
            allowed_decisions=["approve", "reject"],
            checkpointer=checkpointer
        )
    """
    if allowed_decisions:
        config: InterruptOnConfig = {"allowed_decisions": allowed_decisions}
        interrupt_on = {name: config for name in tool_names}
    else:
        interrupt_on = {name: True for name in tool_names}

    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on, checkpointer=checkpointer)
