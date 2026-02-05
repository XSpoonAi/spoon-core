"""
Human-in-the-Loop (HITL) System

Provides approval workflows for critical tool executions:
- Tool-level approval configuration with dynamic descriptions
- Multiple approval strategies (approve, edit, reject)
- Batch interrupt/resume support for parallel tool calls
- State preservation for pause/resume via Command pattern
- Integration with checkpointing

Compatible with LangChain DeepAgents HumanInTheLoopMiddleware interface.

Usage:
    from spoon_ai.tools.hitl import HumanInTheLoopMiddleware, InterruptOnConfig

    # Simple configuration
    agent = ToolCallAgent(
        tools=[dangerous_tool],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={
            "delete_file": True,
            "send_email": {"allowed_decisions": ["approve", "reject"]}
        })]
    )

    # With dynamic description function
    def format_delete_description(tool_call, state, runtime):
        return f"Delete file: {tool_call['args'].get('path', 'unknown')}"

    middleware = HumanInTheLoopMiddleware(interrupt_on={
        "delete_file": {
            "allowed_decisions": ["approve", "reject"],
            "description": format_delete_description,
        }
    })

    # Resume after interrupt
    result = agent.invoke(Command(resume={"decisions": [
        {"type": "approve"},
        {"type": "edit", "args": {"path": "/new/path"}},
    ]}), config=config)
"""

import asyncio
import logging
import time
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
    Sequence,
)
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ToolCallRequest,
    ToolCallResult,
    ModelRequest,
    ModelResponse,
)
from spoon_ai.schema import Message, Role

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions - Compatible with LangChain
# ============================================================================

# Type alias for tool call dict (compatible with LangChain ToolCall)
ToolCall = Dict[str, Any]

# Type alias for agent state
AgentState = Dict[str, Any]

# Type alias for runtime
Runtime = Any

# Description function signature: (tool_call, state, runtime) -> str
DescriptionFunction = Callable[[ToolCall, AgentState, Runtime], str]


def _get_tool_attr(tool_call: Any, key: str, default: Any = None) -> Any:
    """Safely get attribute from tool_call (dict or Pydantic object).

    Args:
        tool_call: Tool call object (dict or Pydantic model)
        key: Attribute name to get
        default: Default value if attribute not found

    Returns:
        The attribute value or default
    """
    # Try dict-style access first
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)

    # Try Pydantic/object attribute access
    if hasattr(tool_call, key):
        value = getattr(tool_call, key, default)
        return value if value is not None else default

    # Try model_dump for Pydantic v2
    if hasattr(tool_call, 'model_dump'):
        return tool_call.model_dump().get(key, default)

    # Try dict() for Pydantic v1
    if hasattr(tool_call, 'dict'):
        return tool_call.dict().get(key, default)

    return default


class InterruptOnConfig(TypedDict, total=False):
    """Configuration for tool interruption.

    Compatible with LangChain DeepAgents InterruptOnConfig.

    Attributes:
        allowed_decisions: List of allowed approval decisions.
            Defaults to ["approve", "edit", "reject"].
        description: Either a static string or a callable that generates
            a dynamic description based on the tool call context.
            Signature: (tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str

    Example:
        # Static description
        config: InterruptOnConfig = {
            "allowed_decisions": ["approve", "reject"],
            "description": "This action will delete files permanently.",
        }

        # Dynamic description
        def format_description(tool_call, state, runtime):
            path = tool_call["args"].get("path", "unknown")
            return f"Delete file: {path}"

        config: InterruptOnConfig = {
            "allowed_decisions": ["approve", "reject"],
            "description": format_description,
        }
    """

    allowed_decisions: List[Literal["approve", "edit", "reject"]]
    description: Union[str, DescriptionFunction]


# ============================================================================
# Approval Decision Types
# ============================================================================

class ApprovalDecision(str, Enum):
    """Possible approval decisions."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


@dataclass
class ApprovalResponse:
    """Response from approval_callback, supporting EDIT with modified arguments.
    
    Usage:
        # Simple approve/reject
        return ApprovalDecision.APPROVE
        
        # Edit with modified arguments
        return ApprovalResponse(
            decision=ApprovalDecision.EDIT,
            modified_arguments={"path": "/new/path", "mode": "read"}
        )
    """
    decision: ApprovalDecision
    modified_arguments: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate that EDIT decisions include modified_arguments."""
        if self.decision == ApprovalDecision.EDIT and self.modified_arguments is None:
            raise ValueError(
                "EDIT decision requires modified_arguments. "
                "Provide modified_arguments or use APPROVE/REJECT instead."
            )


def normalize_decision(decision: Union[ApprovalDecision, str, Any]) -> tuple[ApprovalDecision, Optional[Dict[str, Any]]]:
    """Normalize approval decision from various input types.
    
    Args:
        decision: Can be:
            - ApprovalDecision enum
            - str ("approve", "edit", "reject")
            - Duck-typed ApprovalResponse object (with decision and optional modified_arguments attributes)
              This allows cross-module compatibility without requiring exact class match.
    
    Returns:
        Tuple of (ApprovalDecision, modified_arguments)
        - modified_arguments is None unless decision is EDIT and provided in ApprovalResponse
    
    Examples:
        >>> normalize_decision(ApprovalDecision.APPROVE)
        (ApprovalDecision.APPROVE, None)
        
        >>> normalize_decision("approve")
        (ApprovalDecision.APPROVE, None)
        
        >>> normalize_decision(ApprovalResponse(decision=ApprovalDecision.EDIT, modified_arguments={"x": 1}))
        (ApprovalDecision.EDIT, {"x": 1})
        
        >>> # Duck-typed object (cross-module compatibility)
        >>> class CustomResponse:
        ...     decision = ApprovalDecision.EDIT
        ...     modified_arguments = {"x": 1}
        >>> normalize_decision(CustomResponse())
        (ApprovalDecision.EDIT, {"x": 1})
    """
    # Handle duck-typed ApprovalResponse (check for decision and modified_arguments attributes)
    # This avoids cross-module class mismatch issues
    if hasattr(decision, 'decision') and hasattr(decision, 'modified_arguments'):
        try:
            decision_attr = getattr(decision, 'decision')
            modified_arguments = getattr(decision, 'modified_arguments')
            # Validate that decision attribute is an ApprovalDecision
            if isinstance(decision_attr, ApprovalDecision):
                return (decision_attr, modified_arguments)
            # If it's a string, try to convert it
            elif isinstance(decision_attr, str):
                enum_decision = ApprovalDecision(decision_attr.lower())
                return (enum_decision, modified_arguments)
        except (AttributeError, ValueError):
            # If accessing attributes fails or conversion fails, fall through to other handlers
            pass
    
    # Handle ApprovalDecision enum
    if isinstance(decision, ApprovalDecision):
        return (decision, None)
    
    # Handle string
    if isinstance(decision, str):
        try:
            enum_decision = ApprovalDecision(decision.lower())
            return (enum_decision, None)
        except ValueError:
            raise ValueError(
                f"Invalid approval decision string: {decision}. "
                f"Must be one of: {[d.value for d in ApprovalDecision]}"
            )
    
    # Unknown type
    raise TypeError(
        f"Invalid approval decision type: {type(decision)}. "
        f"Must be ApprovalDecision, str, or ApprovalResponse"
    )


@dataclass
class DecisionInput:
    """Input for a single approval decision.

    Used in Command(resume={"decisions": [...]}) pattern.
    """

    type: Literal["approve", "edit", "reject"]
    args: Optional[Dict[str, Any]] = None  # For edit decision
    reason: Optional[str] = None  # For reject decision

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionInput":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "approve"),
            args=data.get("args"),
            reason=data.get("reason"),
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
class ApprovalRequest:
    """Request object passed to approval_callback.

    Provides a simple interface for approval callbacks.
    """

    tool_name: str
    arguments: Dict[str, Any]
    tool_call_id: str

    @classmethod
    def from_action_request(cls, action: ActionRequest) -> "ApprovalRequest":
        """Create from ActionRequest."""
        return cls(
            tool_name=action.name,
            arguments=action.args,
            tool_call_id=action.id,
        )


@dataclass
class ReviewConfig:
    """Review configuration for a pending action.

    Compatible with LangChain's review_config format.
    """

    action_name: str
    action_id: str
    allowed_decisions: List[str]
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "action_name": self.action_name,
            "action_id": self.action_id,
            "allowed_decisions": self.allowed_decisions,
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class InterruptValue:
    """Value returned in __interrupt__ for batch interrupts.

    Compatible with LangChain's interrupt value format.
    Contains both action_requests and review_configs.
    """

    action_requests: List[ActionRequest]
    review_configs: List[ReviewConfig]

    def __len__(self) -> int:
        """Return number of pending actions (for compatibility)."""
        return 2  # action_requests and review_configs

    def __iter__(self):
        """Iterate over keys (for compatibility)."""
        yield "action_requests"
        yield "review_configs"

    def __getitem__(self, key: str) -> Any:
        """Get item by key (for compatibility)."""
        if key == "action_requests":
            return [r.to_dict() for r in self.action_requests]
        elif key == "review_configs":
            return [c.to_dict() for c in self.review_configs]
        raise KeyError(key)

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
# Command Class for Resume
# ============================================================================

@dataclass
class ResumeData:
    """Data for resuming from an interrupt.

    Compatible with LangChain Command(resume=...) pattern.
    """

    decisions: List[DecisionInput]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeData":
        """Create from dictionary."""
        decisions = [
            DecisionInput.from_dict(d) if isinstance(d, dict) else d
            for d in data.get("decisions", [])
        ]
        return cls(decisions=decisions)


# ============================================================================
# HITL Interrupt Exception
# ============================================================================

class HITLInterrupt(Exception):
    """Exception raised when tool execution requires batch approval.

    This signals to the agent that execution should pause and return
    an __interrupt__ with action_requests and review_configs.
    """

    def __init__(self, interrupt_info: InterruptInfo):
        self.interrupt_info = interrupt_info
        self.value = interrupt_info.value
        super().__init__(f"HITL interrupt: {len(interrupt_info.value.action_requests)} actions pending")


# ============================================================================
# Parsed Config Helper
# ============================================================================

@dataclass
class ParsedInterruptConfig:
    """Parsed and normalized interrupt configuration."""

    allowed_decisions: List[ApprovalDecision]
    description: Optional[Union[str, DescriptionFunction]] = None

    @classmethod
    def from_config(
        cls, 
        config: Union[bool, InterruptOnConfig],
        tool_name: Optional[str] = None,
        strict: bool = True
    ) -> Optional["ParsedInterruptConfig"]:
        """Parse configuration from various formats.
        
        Args:
            config: Configuration (bool or InterruptOnConfig dict)
            tool_name: Optional tool name for error messages
            strict: If True, raise ValueError on invalid allowed_decisions.
                   If False, log warning and use defaults for invalid values.
        
        Returns:
            ParsedInterruptConfig or None if config is False/None
        
        Raises:
            ValueError: If strict=True and allowed_decisions contains invalid values
        """
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
            
            # Validate and normalize allowed_decisions (case-insensitive)
            valid_decisions = []
            invalid_values = []
            valid_decision_values = {d.value.lower(): d for d in ApprovalDecision}
            
            for decision_str in allowed:
                if isinstance(decision_str, ApprovalDecision):
                    # Already an enum, use directly
                    valid_decisions.append(decision_str)
                elif isinstance(decision_str, str):
                    # Case-insensitive lookup
                    normalized = decision_str.lower()
                    if normalized in valid_decision_values:
                        valid_decisions.append(valid_decision_values[normalized])
                    else:
                        invalid_values.append(decision_str)
                else:
                    invalid_values.append(str(decision_str))
            
            # If there are invalid values, handle based on strict mode
            if invalid_values:
                tool_context = f" for tool '{tool_name}'" if tool_name else ""
                valid_options = ", ".join([d.value for d in ApprovalDecision])
                error_msg = (
                    f"Invalid allowed_decisions{tool_context}: {invalid_values}. "
                    f"Valid options are: {valid_options} (case-insensitive)"
                )
                
                if strict:
                    raise ValueError(error_msg)
                else:
                    # Non-strict: log warning and continue with valid decisions
                    logger.warning(
                        f"{error_msg}. Using valid decisions: {[d.value for d in valid_decisions]}"
                    )
            
            # If no valid decisions, use defaults
            if not valid_decisions:
                logger.warning(
                    f"No valid allowed_decisions found{tool_name and f' for tool {tool_name}' or ''}, "
                    f"using defaults: approve, edit, reject"
                )
                valid_decisions = [
                    ApprovalDecision.APPROVE,
                    ApprovalDecision.EDIT,
                    ApprovalDecision.REJECT,
                ]
            
            return cls(
                allowed_decisions=valid_decisions,
                description=config.get("description"),
            )

        return None

    def get_description(
        self,
        tool_call: ToolCall,
        state: AgentState,
        runtime: Runtime,
    ) -> Optional[str]:
        """Get description, calling function if needed."""
        if self.description is None:
            return None
        if callable(self.description):
            try:
                return self.description(tool_call, state, runtime)
            except Exception as e:
                logger.warning(f"Description function failed: {e}")
                return None
        return str(self.description)


# ============================================================================
# HITL Manager
# ============================================================================

class HITLManager:
    """Manages human-in-the-loop approval workflows.

    Supports batch interrupts for parallel tool calls.
    """

    def __init__(
        self,
        interrupt_on: Dict[str, Union[bool, InterruptOnConfig]],
        strict: bool = True,
    ):
        """Initialize HITL manager.

        Args:
            interrupt_on: Tool interruption configuration.
                - Dict[str, bool]: Simple approval (True = require approval)
                - Dict[str, InterruptOnConfig]: Detailed configuration with
                  allowed_decisions and description
            strict: If True, raise ValueError on invalid allowed_decisions.
                   If False, log warning and use defaults for invalid values.
        """
        self.interrupt_on = interrupt_on

        # Parse configurations
        self.tool_configs: Dict[str, ParsedInterruptConfig] = {}
        for tool_name, config in interrupt_on.items():
            parsed = ParsedInterruptConfig.from_config(config, tool_name=tool_name, strict=strict)
            if parsed:
                self.tool_configs[tool_name] = parsed

        # Pending actions for batch interrupt
        self._pending_actions: List[ActionRequest] = []
        self._pending_configs: List[ReviewConfig] = []

        logger.info(f"HITL Manager initialized with {len(self.tool_configs)} tool configurations")

    def should_interrupt(self, tool_name: str) -> bool:
        """Check if tool requires approval."""
        return tool_name in self.tool_configs

    def get_config(self, tool_name: str) -> Optional[ParsedInterruptConfig]:
        """Get parsed configuration for a tool."""
        return self.tool_configs.get(tool_name)

    def add_pending_action(
        self,
        tool_call: ToolCall,
        state: AgentState,
        runtime: Runtime,
    ) -> None:
        """Add a pending action for batch interrupt.

        Args:
            tool_call: The tool call dict or Pydantic object with name, args, id
            state: Current agent state
            runtime: Agent runtime
        """
        tool_name = _get_tool_attr(tool_call, "name", "")
        tool_id = _get_tool_attr(tool_call, "id", str(uuid.uuid4()))
        args = _get_tool_attr(tool_call, "args", {})

        config = self.tool_configs.get(tool_name)
        if not config:
            return

        # Create action request
        action = ActionRequest(
            name=tool_name,
            args=args,
            id=tool_id,
        )
        self._pending_actions.append(action)

        # Create review config with dynamic description
        description = config.get_description(tool_call, state, runtime)
        review = ReviewConfig(
            action_name=tool_name,
            action_id=tool_id,
            allowed_decisions=[d.value for d in config.allowed_decisions],
            description=description,
        )
        self._pending_configs.append(review)

        logger.debug(f"Added pending action: {tool_name} (ID: {tool_id})")

    def has_pending_actions(self) -> bool:
        """Check if there are pending actions."""
        return len(self._pending_actions) > 0

    def create_interrupt(self) -> InterruptInfo:
        """Create an interrupt with all pending actions.

        Returns:
            InterruptInfo with action_requests and review_configs
        """
        interrupt_value = InterruptValue(
            action_requests=self._pending_actions.copy(),
            review_configs=self._pending_configs.copy(),
        )

        interrupt_info = InterruptInfo(value=interrupt_value)

        logger.info(f"Created HITL interrupt with {len(self._pending_actions)} actions")

        return interrupt_info

    def clear_pending(self) -> None:
        """Clear all pending actions."""
        self._pending_actions.clear()
        self._pending_configs.clear()

    def apply_decisions(
        self,
        decisions: List[DecisionInput],
        tool_calls: List[ToolCall],
    ) -> List[ToolCall]:
        """Apply decisions to tool calls.

        Args:
            decisions: List of decisions from Command(resume=...)
            tool_calls: Original tool calls

        Returns:
            Modified tool calls with edits applied, rejected calls removed
        """
        result = []

        for i, (decision, tool_call) in enumerate(zip(decisions, tool_calls)):
            tool_name = _get_tool_attr(tool_call, 'name', 'unknown')
            if decision.type == "reject":
                logger.info(f"Tool {tool_name} rejected: {decision.reason}")
                continue
            elif decision.type == "edit":
                # Apply edited arguments
                modified = deepcopy(tool_call)
                if decision.args:
                    # Handle both dict and Pydantic objects
                    if isinstance(modified, dict):
                        modified["args"] = decision.args
                    elif hasattr(modified, 'args'):
                        modified.args = decision.args
                result.append(modified)
                logger.info(f"Tool {tool_name} approved with edits")
            else:  # approve
                result.append(tool_call)
                logger.info(f"Tool {tool_name} approved")

        return result


# ============================================================================
# Human-in-the-Loop Middleware (LangChain Compatible Name)
# ============================================================================

class HumanInTheLoopMiddleware(AgentMiddleware):
    """Middleware that implements Human-in-the-Loop approval workflows.

    Compatible with LangChain DeepAgents HumanInTheLoopMiddleware.

    This middleware intercepts tool calls that require approval and creates
    an __interrupt__ with action_requests and review_configs for batch approval.

    Features:
    - Per-tool approval configuration
    - Dynamic description functions
    - Batch interrupt/resume for parallel tool calls
    - Command(resume=...) pattern for resuming

    Usage:
        # Simple approval
        middleware = HumanInTheLoopMiddleware(interrupt_on={
            "delete_file": True,
            "send_email": True
        })

        # With dynamic description
        def format_shell_description(tool_call, state, runtime):
            command = tool_call["args"].get("command", "N/A")
            return f"Execute Command: {command}"

        middleware = HumanInTheLoopMiddleware(interrupt_on={
            "shell": {
                "allowed_decisions": ["approve", "reject"],
                "description": format_shell_description,
            },
            "write_file": {
                "allowed_decisions": ["approve", "edit", "reject"],
            },
        })

        # Resume from interrupt
        result = agent.invoke(
            Command(resume={"decisions": [
                {"type": "approve"},
                {"type": "edit", "args": {"path": "/new/path"}},
            ]}),
            config=config
        )

    Interrupt Format (returned in result["__interrupt__"]):
        [
            {
                "value": {
                    "action_requests": [
                        {"name": "shell", "args": {"command": "rm -rf"}, "id": "..."},
                    ],
                    "review_configs": [
                        {
                            "action_name": "shell",
                            "action_id": "...",
                            "allowed_decisions": ["approve", "reject"],
                            "description": "Execute Command: rm -rf",
                        },
                    ],
                },
                "interrupt_id": "...",
            }
        ]
    """

    system_prompt = """# Human-in-the-Loop Approval

Some tools require human approval before execution. When you attempt to use
these tools, execution will pause and return an interrupt with pending actions.

Approval workflow:
1. Tool calls are collected
2. Interrupt is raised with action_requests and review_configs
3. User provides decisions (approve/edit/reject) for each action
4. Execution resumes with Command(resume={"decisions": [...]})

Be prepared for tool calls to be rejected or modified.
"""

    def __init__(
        self,
        interrupt_on: Dict[str, Union[bool, InterruptOnConfig]],
        approval_callback: Optional[Callable[["ApprovalRequest"], Union[ApprovalDecision, "ApprovalResponse"]]] = None,
        strict: bool = True,
    ):
        """Initialize HITL middleware.

        Args:
            interrupt_on: Tool interruption configuration.
                Keys are tool names, values can be:
                - True: Require approval with default allowed_decisions
                - False: No approval needed (tool is skipped)
                - InterruptOnConfig dict with:
                    - allowed_decisions: List of ["approve", "edit", "reject"]
                    - description: Static string or callable for dynamic description
            approval_callback: Optional callback function for automatic approval.
                If provided, this callback is called instead of raising HITLInterrupt.
                The callback receives an ApprovalRequest and returns:
                - ApprovalDecision (APPROVE/REJECT)
                - ApprovalResponse (for EDIT with modified arguments)
                
                Examples:
                    # Simple approve/reject
                    def auto_approve(request):
                        if request.tool_name == "dangerous_tool":
                            return ApprovalDecision.REJECT
                        return ApprovalDecision.APPROVE
                    
                    # Edit with modified arguments
                    def auto_edit(request):
                        if request.tool_name == "file_write":
                            return ApprovalResponse(
                                decision=ApprovalDecision.EDIT,
                                modified_arguments={"path": "/safe/path", **request.arguments}
                            )
                        return ApprovalDecision.APPROVE
            strict: If True, raise ValueError on invalid allowed_decisions.
                   If False, log warning and use defaults for invalid values.
                   Defaults to True for strict validation.
        """
        super().__init__()
        self.interrupt_on = interrupt_on
        self.approval_callback = approval_callback
        self.manager = HITLManager(interrupt_on, strict=strict)

        # Track pending resume data
        self._resume_data: Optional[ResumeData] = None
        
        # Cache approval decisions by tool_call_id for idempotency
        # This ensures that retries use the same decision without re-calling the callback
        self._approval_cache: Dict[str, tuple[ApprovalDecision, Optional[Dict[str, Any]]]] = {}

        logger.info(f"HumanInTheLoopMiddleware initialized for {len(self.manager.tool_configs)} tools")

    def set_resume_data(self, resume: Dict[str, Any]) -> None:
        """Set resume data from Command(resume=...).

        Called by the agent when resuming from interrupt.
        """
        self._resume_data = ResumeData.from_dict(resume)
        logger.info(f"Resume data set with {len(self._resume_data.decisions)} decisions")

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime,
    ) -> Optional[Dict[str, Any]]:
        """Initialize state for HITL tracking."""
        # Clear any previous pending actions
        self.manager.clear_pending()
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Intercept model response to collect tool calls requiring approval.

        This processes the model response and identifies tool calls that
        need approval, then raises an HITLInterrupt if any are found.
        """
        # Call the model
        response = await handler(request)

        # Check if there are tool calls requiring approval
        if not response.tool_calls:
            return response

        # Get state and runtime for description functions
        state = {}
        runtime_instance = request.runtime
        if runtime_instance and hasattr(runtime_instance, '_agent_instance'):
            agent = runtime_instance._agent_instance
            if hasattr(agent, '_agent_state'):
                state = agent._agent_state

        # Check if we're resuming from interrupt
        if self._resume_data:
            # Apply decisions to tool calls
            decisions = self._resume_data.decisions
            self._resume_data = None

            # Filter and modify tool calls based on decisions
            approved_calls = []
            rejection_messages = []

            for i, tool_call in enumerate(response.tool_calls):
                if i < len(decisions):
                    decision = decisions[i]
                    if decision.type == "reject":
                        rejection_messages.append(
                            f"Tool '{_get_tool_attr(tool_call, 'name')}' was rejected: {decision.reason or 'User rejected'}"
                        )
                    elif decision.type == "edit":
                        modified = deepcopy(tool_call)
                        if decision.args:
                            # Handle both dict and Pydantic objects
                            if isinstance(modified, dict):
                                modified["args"] = decision.args
                            elif hasattr(modified, 'args'):
                                modified.args = decision.args
                        approved_calls.append(modified)
                    else:
                        approved_calls.append(tool_call)
                else:
                    approved_calls.append(tool_call)

            # Update response with approved calls
            response.tool_calls = approved_calls

            # Log rejections and add to response content so agent knows
            if rejection_messages:
                rejection_text = "\n".join(rejection_messages)
                logger.info(f"Tools rejected: {rejection_text}")
                
                # Add rejection message to response content so agent is informed
                current_content = response.content or ""
                if current_content:
                    response.content = f"{current_content}\n\n{rejection_text}"
                else:
                    response.content = rejection_text

            return response

        # Collect tool calls requiring approval
        tools_requiring_approval = []
        tools_not_requiring_approval = []

        for tool_call in response.tool_calls:
            tool_name = _get_tool_attr(tool_call, "name", "")
            if self.manager.should_interrupt(tool_name):
                tools_requiring_approval.append(tool_call)
                # Only add to pending queue for manual interrupt/resume (no callback mode)
                # If approval_callback exists, let awrap_tool_call handle it directly
                if not self.approval_callback:
                    self.manager.add_pending_action(
                        tool_call,
                        state,
                        runtime_instance,
                    )
            else:
                tools_not_requiring_approval.append(tool_call)

        # If there are tools requiring approval
        if self.manager.has_pending_actions():
            # NOTE: approval_callback is handled in awrap_tool_call (single authoritative interception point)
            # This prevents callback from being called twice and ensures consistent decisions.
            # Only raise interrupt for manual approval (no callback mode)
            interrupt_info = self.manager.create_interrupt()
            self.manager.clear_pending()
            raise HITLInterrupt(interrupt_info)
        
        # If approval_callback exists but no pending actions (tools were added above only if no callback),
        # ensure pending queue is cleared to avoid accumulation
        if self.approval_callback:
            self.manager.clear_pending()

        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult],
    ) -> ToolCallResult:
        """Intercept tool execution to enforce approval decisions.
        
        This is the PRIMARY interception point for tool execution in ToolCallAgent.
        Even if a tool call makes it through awrap_model_call, we check here before
        actual execution.
        
        CRITICAL: This method MUST be called before tool execution, and MUST return
        ToolCallResult(success=False, ...) when rejected to prevent execution.
        """
        # Check if this tool requires approval
        if self.manager.should_interrupt(request.tool_name):
            # If approval_callback is provided, use it for automatic decisions
            if self.approval_callback:
                # Check cache for idempotency (avoid repeated modifications on retries)
                cache_key = request.tool_call_id
                if cache_key in self._approval_cache:
                    # Use cached decision for idempotency
                    decision, modified_arguments = self._approval_cache[cache_key]
                    logger.debug(
                        f"Tool '{request.tool_name}' (ID: {cache_key}) using cached approval decision: {decision.value}"
                    )
                else:
                    # First time: call callback and cache the result
                    approval_request = ApprovalRequest(
                        tool_name=request.tool_name,
                        arguments=request.arguments,
                        tool_call_id=request.tool_call_id,
                    )
                    
                    callback_result = self.approval_callback(approval_request)
                    
                    # Normalize decision (handles ApprovalDecision, str, ApprovalResponse)
                    # Wrap with try/except for stability
                    try:
                        decision, modified_arguments = normalize_decision(callback_result)
                    except (ValueError, TypeError) as e:
                        error_msg = f"Tool '{request.tool_name}' approval callback returned invalid decision: {e}"
                        logger.error(error_msg)
                        return ToolCallResult.from_error(error_msg)
                    
                    # Cache the decision for idempotency
                    self._approval_cache[cache_key] = (decision, modified_arguments)
                    logger.debug(
                        f"Tool '{request.tool_name}' (ID: {cache_key}) approval decision cached: {decision.value}"
                    )
                
                # Enforce allowed_decisions from InterruptOnConfig
                config = self.manager.get_config(request.tool_name)
                if config and config.allowed_decisions:
                    if decision not in config.allowed_decisions:
                        allowed_str = ", ".join([d.value for d in config.allowed_decisions])
                        error_msg = (
                            f"Tool '{request.tool_name}' approval decision '{decision.value}' "
                            f"is not in allowed_decisions: [{allowed_str}]. "
                            f"Callback must return one of: {allowed_str}"
                        )
                        logger.warning(error_msg)
                        return ToolCallResult.from_error(error_msg)
                
                if decision == ApprovalDecision.REJECT:
                    # Tool was rejected - return error result to prevent execution
                    error_msg = f"Tool '{request.tool_name}' was rejected by approval callback"
                    logger.warning(error_msg)
                    # Return ToolCallResult with success=False (error set)
                    return ToolCallResult.from_error(error_msg)
                elif decision == ApprovalDecision.APPROVE:
                    # Tool was approved - proceed with execution
                    logger.debug(f"Tool '{request.tool_name}' approved, proceeding with execution")
                    return await handler(request)
                elif decision == ApprovalDecision.EDIT:
                    # EDIT requires modified arguments
                    if modified_arguments is None:
                        error_msg = (
                            f"Tool '{request.tool_name}' approval decision EDIT requires modified_arguments. "
                            "Return ApprovalResponse(decision=EDIT, modified_arguments={{...}}) from callback."
                        )
                        logger.warning(error_msg)
                        return ToolCallResult.from_error(error_msg)
                    
                    # Create new request with modified arguments (from cache if retry)
                    # This ensures idempotency: same tool_call_id always uses same modified args
                    modified_request = ToolCallRequest(
                        tool_name=request.tool_name,
                        arguments=modified_arguments,  # Always use cached/modified args
                        tool_call_id=request.tool_call_id,
                        runtime=request.runtime,
                        tool_call=request.tool_call,
                    )
                    logger.info(
                        f"Tool '{request.tool_name}' approved with EDIT (idempotent). "
                        f"Request args: {request.arguments}, Using modified args: {modified_arguments}"
                    )
                    return await handler(modified_request)
                else:
                    # Unknown decision type
                    logger.warning(f"Unknown approval decision: {decision}, rejecting for safety")
                    return ToolCallResult.from_error(
                        f"Tool '{request.tool_name}' approval decision invalid: {decision}"
                    )
            else:
                # No callback - should have been handled in awrap_model_call
                # But if we get here without callback, we need to raise interrupt
                logger.warning(f"Tool '{request.tool_name}' requires approval but no callback provided, raising interrupt")
                # Create interrupt for manual approval
                state = {}
                runtime = request.runtime
                if runtime and hasattr(runtime, '_agent_instance'):
                    agent = runtime._agent_instance
                    if hasattr(agent, '_agent_state'):
                        state = agent._agent_state
                
                self.manager.add_pending_action(
                    request.tool_call or {"name": request.tool_name, "args": request.arguments},
                    state,
                    runtime,
                )
                
                if self.manager.has_pending_actions():
                    interrupt_info = self.manager.create_interrupt()
                    self.manager.clear_pending()
                    raise HITLInterrupt(interrupt_info)
        
        # Tool doesn't require approval - proceed with execution
        return await handler(request)

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

# Alias for backward compatibility with existing code
HITLMiddleware = HumanInTheLoopMiddleware


# ============================================================================
# Convenience Functions
# ============================================================================

def create_hitl_middleware(
    *tool_names: str,
    allowed_decisions: Optional[List[str]] = None,
) -> HumanInTheLoopMiddleware:
    """Create HITL middleware for specified tools.

    Args:
        *tool_names: Names of tools that require approval
        allowed_decisions: Allowed decisions for all tools

    Returns:
        Configured HumanInTheLoopMiddleware

    Example:
        middleware = create_hitl_middleware(
            "delete_file",
            "send_email",
            "shutdown_server",
            allowed_decisions=["approve", "reject"]
        )
    """
    if allowed_decisions:
        config: InterruptOnConfig = {"allowed_decisions": allowed_decisions}
        interrupt_on = {name: config for name in tool_names}
    else:
        interrupt_on = {name: True for name in tool_names}

    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on)


def format_tool_call_description(
    tool_call: ToolCall,
    state: AgentState,
    runtime: Runtime,
) -> str:
    """Default description formatter for tool calls.

    Can be used as a base for custom description functions.
    """
    name = _get_tool_attr(tool_call, "name", "unknown")
    args = _get_tool_attr(tool_call, "args", {})

    # Format arguments
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())

    return f"{name}({args_str})"
