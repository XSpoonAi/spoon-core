"""
Human-in-the-Loop (HITL) System

Provides approval workflows for critical tool executions:
- Tool-level approval configuration
- Multiple approval strategies (approve, edit, reject)
- State preservation for pause/resume
- Integration with checkpointing

Usage:
    agent = ToolCallAgent(
        tools=[dangerous_tool],
        middleware=[HITLMiddleware(interrupt_on={
            "delete_file": True,
            "send_email": {"allowed_decisions": ["approve", "reject"]}
        })]
    )
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Literal, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from spoon_ai.middleware.base import (
    AgentMiddleware,
    ToolCallRequest,
    ToolCallResult,
    AgentRuntime
)

logger = logging.getLogger(__name__)


# ============================================================================
# HITL Configuration Types
# ============================================================================

class ApprovalDecision(str, Enum):
    """Possible approval decisions."""
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


@dataclass
class InterruptConfig:
    """Configuration for tool interruption."""
    allowed_decisions: List[ApprovalDecision] = field(default_factory=lambda: [
        ApprovalDecision.APPROVE,
        ApprovalDecision.EDIT,
        ApprovalDecision.REJECT
    ])
    approval_message: Optional[str] = None
    auto_approve_after: Optional[int] = None  # Seconds before auto-approval


@dataclass
class ToolApprovalRequest:
    """A pending tool approval request."""
    # Tool identification
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]

    # Configuration
    config: InterruptConfig

    # Status
    decision: Optional[ApprovalDecision] = None
    edited_arguments: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None

    # Metadata
    created_at: float = field(default_factory=lambda: __import__('time').time())

    def is_approved(self) -> bool:
        """Check if request is approved."""
        return self.decision == ApprovalDecision.APPROVE

    def is_rejected(self) -> bool:
        """Check if request is rejected."""
        return self.decision == ApprovalDecision.REJECT

    def is_edited(self) -> bool:
        """Check if request has edits."""
        return self.decision == ApprovalDecision.EDIT

    def get_final_arguments(self) -> Dict[str, Any]:
        """Get final arguments (original or edited)."""
        if self.is_edited() and self.edited_arguments:
            return self.edited_arguments
        return self.arguments


# ============================================================================
# HITL Exception
# ============================================================================

class ToolApprovalRequired(Exception):
    """Exception raised when tool execution requires approval.

    This signals to the agent that execution should pause and wait
    for user approval before proceeding.
    """

    def __init__(self, approval_request: ToolApprovalRequest):
        self.approval_request = approval_request
        super().__init__(f"Tool '{approval_request.tool_name}' requires approval")


# ============================================================================
# HITL Manager
# ============================================================================

class HITLManager:
    """Manages human-in-the-loop approval workflows."""

    def __init__(
        self,
        interrupt_on: Dict[str, Any],
        approval_callback: Optional[Callable[[ToolApprovalRequest], ApprovalDecision]] = None
    ):
        """Initialize HITL manager.

        Args:
            interrupt_on: Tool interruption configuration
                - Dict[str, bool]: Simple approval (True = require approval)
                - Dict[str, InterruptConfig]: Detailed configuration
            approval_callback: Optional callback for programmatic approval
        """
        self.interrupt_on = interrupt_on
        self.approval_callback = approval_callback

        # Parse configuration
        self.tool_configs: Dict[str, InterruptConfig] = {}
        for tool_name, config in interrupt_on.items():
            if isinstance(config, bool):
                if config:
                    self.tool_configs[tool_name] = InterruptConfig()
            elif isinstance(config, dict):
                self.tool_configs[tool_name] = InterruptConfig(**config)
            elif isinstance(config, InterruptConfig):
                self.tool_configs[tool_name] = config

        # Pending approvals
        self.pending_approvals: List[ToolApprovalRequest] = []
        self._approval_event = asyncio.Event()

        logger.info(f"HITL Manager initialized with {len(self.tool_configs)} tool configurations")

    def should_interrupt(self, tool_name: str) -> bool:
        """Check if tool requires approval.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool requires approval
        """
        return tool_name in self.tool_configs

    def create_approval_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str
    ) -> ToolApprovalRequest:
        """Create an approval request for a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            tool_call_id: Unique tool call ID

        Returns:
            ToolApprovalRequest instance
        """
        config = self.tool_configs.get(tool_name, InterruptConfig())

        request = ToolApprovalRequest(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            config=config
        )

        self.pending_approvals.append(request)
        logger.info(f"Created approval request for tool '{tool_name}' (ID: {tool_call_id})")

        return request

    async def wait_for_approval(
        self,
        request: ToolApprovalRequest,
        timeout: Optional[float] = None
    ) -> ToolApprovalRequest:
        """Wait for user approval of a tool call.

        Args:
            request: The approval request
            timeout: Optional timeout in seconds

        Returns:
            Updated approval request with decision

        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        # Check for approval callback
        if self.approval_callback:
            decision = self.approval_callback(request)
            request.decision = decision
            logger.info(f"Tool '{request.tool_name}' decision via callback: {decision.value}")
            return request

        # Wait for manual approval
        logger.info(f"Waiting for approval of tool '{request.tool_name}'...")

        # Set event when decision is made
        try:
            if timeout:
                await asyncio.wait_for(self._approval_event.wait(), timeout=timeout)
            else:
                await self._approval_event.wait()
        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout for tool '{request.tool_name}'")
            raise

        return request

    def approve_request(
        self,
        tool_call_id: str,
        decision: ApprovalDecision,
        edited_arguments: Optional[Dict[str, Any]] = None,
        rejection_reason: Optional[str] = None
    ) -> bool:
        """Approve, edit, or reject a pending request.

        Args:
            tool_call_id: ID of the tool call
            decision: Approval decision
            edited_arguments: Modified arguments (for EDIT decision)
            rejection_reason: Reason for rejection (for REJECT decision)

        Returns:
            True if request was found and updated
        """
        for request in self.pending_approvals:
            if request.tool_call_id == tool_call_id:
                request.decision = decision
                request.edited_arguments = edited_arguments
                request.rejection_reason = rejection_reason

                logger.info(f"Tool '{request.tool_name}' approved with decision: {decision.value}")

                # Signal approval event
                self._approval_event.set()
                return True

        logger.warning(f"No pending approval found for tool call ID: {tool_call_id}")
        return False

    def get_pending_approvals(self) -> List[ToolApprovalRequest]:
        """Get all pending approval requests.

        Returns:
            List of pending approval requests
        """
        return [r for r in self.pending_approvals if r.decision is None]

    def clear_pending(self) -> None:
        """Clear all pending approvals."""
        self.pending_approvals.clear()
        self._approval_event.clear()


# ============================================================================
# HITL Middleware
# ============================================================================

class HITLMiddleware(AgentMiddleware):
    """Middleware that implements Human-in-the-Loop approval workflows.

    This middleware intercepts tool calls that require approval and pauses
    execution until the user provides a decision (approve, edit, reject).

    Features:
    - Per-tool approval configuration
    - Multiple approval strategies
    - Programmatic approval callbacks
    - State preservation for pause/resume

    Usage:
        # Simple approval (approve/edit/reject)
        middleware = HITLMiddleware(interrupt_on={
            "delete_file": True,
            "send_email": True
        })

        # Custom configuration
        middleware = HITLMiddleware(interrupt_on={
            "dangerous_tool": {
                "allowed_decisions": ["approve", "reject"],  # No edit
                "approval_message": "This tool is dangerous!"
            }
        })

        # Programmatic approval
        def auto_approver(request):
            if request.tool_name == "safe_tool":
                return ApprovalDecision.APPROVE
            return ApprovalDecision.REJECT

        middleware = HITLMiddleware(
            interrupt_on={"dangerous_tool": True},
            approval_callback=auto_approver
        )
    """

    system_prompt = """# Human-in-the-Loop Approval

Some tools require human approval before execution. When you attempt to use
these tools, execution will pause until approval is granted.

Approval workflow:
1. Tool call is intercepted
2. User is prompted for approval
3. User can: approve, edit arguments, or reject
4. Execution resumes based on decision

Be prepared for tool calls to be rejected or modified.
"""

    def __init__(
        self,
        interrupt_on: Dict[str, Any],
        approval_callback: Optional[Callable[[ToolApprovalRequest], ApprovalDecision]] = None,
        approval_timeout: Optional[float] = None
    ):
        """Initialize HITL middleware.

        Args:
            interrupt_on: Tool interruption configuration
            approval_callback: Optional callback for programmatic approval
            approval_timeout: Timeout for approval requests (seconds)
        """
        super().__init__()
        self.manager = HITLManager(interrupt_on, approval_callback)
        self.approval_timeout = approval_timeout

        logger.info(f"HITL Middleware initialized for {len(self.manager.tool_configs)} tools")

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult]
    ) -> ToolCallResult:
        """Intercept tool calls and require approval if configured.

        Args:
            request: Tool call request
            handler: Next handler in chain

        Returns:
            Tool call result (or rejection message)
        """
        # Check if this tool requires approval
        if not self.manager.should_interrupt(request.tool_name):
            # No approval needed, execute normally
            return await handler(request)

        logger.info(f"Tool '{request.tool_name}' requires approval")

        # Create approval request
        approval_request = self.manager.create_approval_request(
            tool_name=request.tool_name,
            arguments=request.arguments,
            tool_call_id=request.tool_call_id
        )

        # Wait for approval
        try:
            approved_request = await self.manager.wait_for_approval(
                approval_request,
                timeout=self.approval_timeout
            )
        except asyncio.TimeoutError:
            return ToolCallResult.from_error(
                f"Tool '{request.tool_name}' approval timed out after {self.approval_timeout}s"
            )

        # Handle decision
        if approved_request.is_rejected():
            reason = approved_request.rejection_reason or "User rejected tool call"
            logger.info(f"Tool '{request.tool_name}' rejected: {reason}")
            return ToolCallResult.from_error(f"Tool call rejected: {reason}")

        if approved_request.is_edited():
            logger.info(f"Tool '{request.tool_name}' approved with edits")
            # Update request with edited arguments
            request.arguments = approved_request.get_final_arguments()
        else:
            logger.info(f"Tool '{request.tool_name}' approved")

        # Execute with final arguments
        return await handler(request)

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get pending approval requests (for UI/CLI).

        Returns:
            List of pending approval dicts
        """
        pending = self.manager.get_pending_approvals()
        return [
            {
                "tool_call_id": req.tool_call_id,
                "tool_name": req.tool_name,
                "arguments": req.arguments,
                "allowed_decisions": [d.value for d in req.config.allowed_decisions],
                "message": req.config.approval_message
            }
            for req in pending
        ]

    def approve(
        self,
        tool_call_id: str,
        decision: Literal["approve", "edit", "reject"],
        edited_arguments: Optional[Dict[str, Any]] = None,
        rejection_reason: Optional[str] = None
    ) -> bool:
        """Approve a pending tool call.

        Args:
            tool_call_id: ID of the tool call
            decision: Approval decision
            edited_arguments: Modified arguments (for edit)
            rejection_reason: Reason for rejection

        Returns:
            True if successful
        """
        decision_enum = ApprovalDecision(decision)
        return self.manager.approve_request(
            tool_call_id,
            decision_enum,
            edited_arguments,
            rejection_reason
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_hitl_middleware(
    *tool_names: str,
    approval_callback: Optional[Callable] = None
) -> HITLMiddleware:
    """Create HITL middleware for specified tools.

    Args:
        *tool_names: Names of tools that require approval
        approval_callback: Optional approval callback

    Returns:
        Configured HITLMiddleware

    Example:
        middleware = create_hitl_middleware(
            "delete_file",
            "send_email",
            "shutdown_server"
        )
    """
    interrupt_on = {name: True for name in tool_names}
    return HITLMiddleware(interrupt_on, approval_callback)
