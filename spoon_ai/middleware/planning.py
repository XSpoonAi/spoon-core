"""
Planning Middleware - Example Plan-Act-Reflect Implementation

This middleware demonstrates how to implement a full Plan-Act-Reflect cycle:
1. PLAN phase: Create a structured plan from the user's request
2. ACT phase: Track execution against the plan
3. REFLECT phase: Evaluate progress and revise the plan if needed

This is a reference implementation showing best practices for Deep Agents.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from spoon_ai.middleware.base import AgentMiddleware, AgentRuntime, AgentPhase
from spoon_ai.tools.base import BaseTool
from spoon_ai.schema import Role

logger = logging.getLogger(__name__)


# ============================================================================
# Plan Data Structures
# ============================================================================

@dataclass
class PlanStep:
    """A single step in the execution plan."""
    id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    """Structured execution plan for a task."""
    goal: str
    steps: List[PlanStep]
    created_at: str
    revised_at: Optional[str] = None
    revision_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
            "revised_at": self.revised_at,
            "revision_count": self.revision_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        steps = [PlanStep(**step) for step in data.get("steps", [])]
        return cls(
            goal=data["goal"],
            steps=steps,
            created_at=data["created_at"],
            revised_at=data.get("revised_at"),
            revision_count=data.get("revision_count", 0),
        )

    def get_pending_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == "pending"]

    def get_completed_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == "completed"]

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the first pending step (current task)."""
        pending = self.get_pending_steps()
        return pending[0] if pending else None

    def mark_step_completed(self, step_id: int, result: str) -> None:
        for step in self.steps:
            if step.id == step_id:
                step.status = "completed"
                step.result = result
                break

    def progress_percentage(self) -> float:
        if not self.steps:
            return 0.0
        completed = len(self.get_completed_steps())
        return (completed / len(self.steps)) * 100


# ============================================================================
# Planning Tools
# ============================================================================

class CreatePlanTool(BaseTool):
    """Tool for the agent to create a structured execution plan."""

    name: str = "create_plan"
    description: str = """Create a detailed step-by-step plan for accomplishing the goal.

    Use this tool to break down complex tasks into manageable steps.
    Each step should be specific, actionable, and measurable.

    Args:
        goal: The overall goal to accomplish
        steps: List of step descriptions (strings)

    Returns:
        Confirmation message with plan summary
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The overall goal of the plan"
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of step descriptions"
            }
        },
        "required": ["goal", "steps"]
    }

    # Pydantic field for middleware reference
    _middleware: Any = None

    def __init__(self, middleware: 'PlanningMiddleware', **kwargs):
        super().__init__(**kwargs)
        # Store middleware reference (not a Pydantic field)
        object.__setattr__(self, '_middleware', middleware)

    async def execute(self, goal: str, steps: List[str], **kwargs) -> str:
        """Create a new execution plan."""
        if isinstance(steps, str):
            # Try to parse as JSON array
            try:
                steps = json.loads(steps)
            except (json.JSONDecodeError, ValueError):
                # If not JSON, split by newlines or treat as single step
                if '\n' in steps:
                    steps = [s.strip() for s in steps.split('\n') if s.strip()]
                else:
                    steps = [steps]

        # Ensure steps is a list
        if not isinstance(steps, list):
            steps = [str(steps)]

        # Filter out empty steps
        steps = [s for s in steps if s and str(s).strip()]

        if not steps:
            return "Error: No valid steps provided. Please provide a list of step descriptions."

        plan_steps = [
            PlanStep(id=i+1, description=str(desc).strip())
            for i, desc in enumerate(steps)
        ]

        plan = ExecutionPlan(
            goal=goal,
            steps=plan_steps,
            created_at=datetime.now().isoformat()
        )

        # Store in middleware
        self._middleware._current_plan = plan

        logger.info(f"Created plan with {len(plan_steps)} steps for goal: {goal}")

        return f"""✓ Plan created with {len(plan_steps)} steps:

Goal: {goal}

Steps:
""" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))


class UpdatePlanTool(BaseTool):
    """Tool for the agent to revise the execution plan."""

    name: str = "update_plan"
    description: str = """Revise the existing plan when circumstances change or the current approach isn't working.

    Use this when:
    - The current plan is not making progress
    - You discovered a better approach
    - Requirements changed

    Args:
        reason: Why the plan needs updating
        new_steps: Revised list of step descriptions

    Returns:
        Confirmation message
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Reason for updating the plan"
            },
            "new_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Revised list of step descriptions"
            }
        },
        "required": ["reason", "new_steps"]
    }

    # Pydantic field for middleware reference
    _middleware: Any = None

    def __init__(self, middleware: 'PlanningMiddleware', **kwargs):
        super().__init__(**kwargs)
        # Store middleware reference (not a Pydantic field)
        object.__setattr__(self, '_middleware', middleware)

    async def execute(self, reason: str, new_steps: List[str], **kwargs) -> str:
        """Update the existing plan."""
        if not self._middleware._current_plan:
            return "Error: No existing plan to update. Use create_plan first."

        if isinstance(new_steps, str):
            # Try to parse as JSON array
            try:
                import json
                new_steps = json.loads(new_steps)
            except (json.JSONDecodeError, ValueError):
                # If not JSON, split by newlines or treat as single step
                if '\n' in new_steps:
                    new_steps = [s.strip() for s in new_steps.split('\n') if s.strip()]
                else:
                    new_steps = [new_steps]

        # Ensure new_steps is a list
        if not isinstance(new_steps, list):
            new_steps = [str(new_steps)]

        # Filter out empty steps
        new_steps = [s for s in new_steps if s and str(s).strip()]

        if not new_steps:
            return "Error: No valid steps provided. Please provide a list of step descriptions."

        # Preserve goal and increment revision count
        old_plan = self._middleware._current_plan

        plan_steps = [
            PlanStep(id=i+1, description=str(desc).strip())
            for i, desc in enumerate(new_steps)
        ]

        new_plan = ExecutionPlan(
            goal=old_plan.goal,
            steps=plan_steps,
            created_at=old_plan.created_at,
            revised_at=datetime.now().isoformat(),
            revision_count=old_plan.revision_count + 1
        )

        self._middleware._current_plan = new_plan

        logger.info(f"Revised plan (revision {new_plan.revision_count}): {reason}")

        return f"""✓ Plan updated (revision {new_plan.revision_count})

Reason: {reason}

New steps:
""" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(new_steps))


# ============================================================================
# Planning Middleware
# ============================================================================

class PlanningMiddleware(AgentMiddleware):
    """Middleware that implements Plan-Act-Reflect for deep reasoning.

    Features:
    - Automatic plan generation in PLAN phase
    - Progress tracking during action loop
    - Periodic reflection and plan revision in REFLECT phase
    - Tools for manual plan creation/updating

    Usage:
        from spoon_ai.middleware.planning import PlanningMiddleware

        agent = ToolCallAgent(
            name="deep-agent",
            middleware=[PlanningMiddleware(auto_plan=True)],
            enable_plan_phase=True,
            enable_reflect_phase=True,
            reflect_interval=3  # Reflect every 3 steps
        )
    """

    # System prompt extension
    system_prompt = """# Planning and Reflection

You are equipped with planning capabilities. Follow this approach:

1. **Plan First**: Break down complex tasks into clear, actionable steps
2. **Execute Methodically**: Follow your plan step-by-step
3. **Reflect Periodically**: Evaluate progress and revise the plan if needed

Available planning tools:
- `create_plan`: Create a detailed execution plan
- `update_plan`: Revise the plan when needed

Best practices:
- Start with a plan for non-trivial tasks
- Each plan step should be specific and measurable
- Reflect on progress every few steps
- Don't hesitate to revise the plan if it's not working
"""

    def __init__(
        self,
        auto_plan: bool = True,
        require_explicit_plan: bool = False,
        enable_progress_tracking: bool = True
    ):
        """Initialize planning middleware.

        Args:
            auto_plan: Automatically generate plan in PLAN phase
            require_explicit_plan: Force agent to create plan before acting
            enable_progress_tracking: Track step completion
        """
        super().__init__()
        self.auto_plan = auto_plan
        self.require_explicit_plan = require_explicit_plan
        self.enable_progress_tracking = enable_progress_tracking

        # Internal state
        self._current_plan: Optional[ExecutionPlan] = None

        # Initialize tools
        self.tools = [
            CreatePlanTool(self),
            UpdatePlanTool(self)
        ]

    # ========================================================================
    # Phase Hooks - Plan-Act-Reflect Implementation
    # ========================================================================

    def on_plan_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate initial plan from user request.

        This is called BEFORE the action loop starts.
        """
        logger.info(f"[PlanningMiddleware] PLAN phase for {runtime.agent_name}")

        user_message = runtime.get_last_message(Role.USER)
        if not user_message:
            logger.warning("No user message found in PLAN phase")
            return None

        user_request = user_message.text_content

        if self.auto_plan:
            # Auto-generate a simple plan structure
            # In practice, you'd use an LLM to generate this
            plan = ExecutionPlan(
                goal=user_request,
                steps=[
                    PlanStep(id=1, description="Analyze the requirements"),
                    PlanStep(id=2, description="Gather necessary information"),
                    PlanStep(id=3, description="Execute the main task"),
                    PlanStep(id=4, description="Verify and finalize results"),
                ],
                created_at=datetime.now().isoformat()
            )

            self._current_plan = plan

            logger.info(f"Auto-generated plan with {len(plan.steps)} steps")

            return {
                "plan": plan.to_dict(),
                "plan_created": True,
                "plan_auto_generated": True
            }

        # Return state to indicate planning is available
        return {
            "plan_available": True,
            "plan_required": self.require_explicit_plan
        }

    def on_think_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Inject plan context into thinking phase."""
        if not self._current_plan:
            return None

        current_step = self._current_plan.get_current_step()
        if not current_step:
            return None

        logger.debug(f"Current plan step: {current_step.description}")

        return {
            "current_plan_step": current_step.to_dict(),
            "plan_progress": self._current_plan.progress_percentage()
        }

    def on_observe_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Track progress after observations."""
        if not self._current_plan or not self.enable_progress_tracking:
            return None

        # Update step status based on observations
        # This is a simplified version - in practice you'd use more sophisticated logic
        current_step = self._current_plan.get_current_step()
        if current_step:
            # Mark as in progress
            current_step.status = "in_progress"

        return None

    def on_reflect_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate progress and potentially revise plan.

        This is called periodically during execution (every N steps).
        """
        logger.info(f"[PlanningMiddleware] REFLECT phase at step {runtime.current_step}")

        if not self._current_plan:
            logger.debug("No plan to reflect on")
            return None

        # Calculate progress
        progress = self._current_plan.progress_percentage()
        completed = len(self._current_plan.get_completed_steps())
        total = len(self._current_plan.steps)

        logger.info(f"Plan progress: {progress:.1f}% ({completed}/{total} steps completed)")

        # Determine if replanning is needed
        # Simple heuristics (in practice, use LLM to evaluate)
        needs_replan = False
        replan_reason = None

        # Check if we're stuck (no progress in recent steps)
        if runtime.current_step > 3 and completed == 0:
            needs_replan = True
            replan_reason = "No progress made in initial steps"

        # Check if we're running out of steps
        steps_remaining = runtime.max_steps - runtime.current_step
        plan_steps_remaining = len(self._current_plan.get_pending_steps())
        if plan_steps_remaining > steps_remaining * 2:
            needs_replan = True
            replan_reason = f"Plan has {plan_steps_remaining} steps but only {steps_remaining} iterations left"

        reflection_data = {
            "plan_progress": progress,
            "completed_steps": completed,
            "total_steps": total,
            "needs_replan": needs_replan,
            "replan_reason": replan_reason,
            "reflection_step": runtime.current_step
        }

        if needs_replan:
            logger.warning(f"Replanning recommended: {replan_reason}")

        return reflection_data

    def on_finish_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Finalize plan and generate summary."""
        logger.info(f"[PlanningMiddleware] FINISH phase")

        if not self._current_plan:
            return None

        # Calculate final statistics
        completed = len(self._current_plan.get_completed_steps())
        total = len(self._current_plan.steps)
        progress = self._current_plan.progress_percentage()

        summary = {
            "plan_goal": self._current_plan.goal,
            "plan_completed": progress == 100.0,
            "steps_completed": completed,
            "total_steps": total,
            "final_progress": progress,
            "plan_revisions": self._current_plan.revision_count
        }

        logger.info(f"Plan execution complete: {progress:.1f}% ({completed}/{total} steps)")

        return summary

    # ========================================================================
    # Agent Lifecycle Hooks
    # ========================================================================

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Initialize planning state."""
        # Restore plan from state if it exists
        if "plan" in state and state["plan"]:
            try:
                self._current_plan = ExecutionPlan.from_dict(state["plan"])
                logger.info("Restored plan from state")
            except Exception as e:
                logger.error(f"Failed to restore plan: {e}")

        return None

    def after_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Save plan to state for persistence."""
        if self._current_plan:
            return {
                "plan": self._current_plan.to_dict()
            }
        return None


# ============================================================================
# Convenience Factory
# ============================================================================

def create_planning_middleware(
    auto_plan: bool = True,
    require_explicit_plan: bool = False
) -> PlanningMiddleware:
    """Create a configured planning middleware instance.

    Args:
        auto_plan: Automatically generate plan in PLAN phase
        require_explicit_plan: Force agent to create plan before acting

    Returns:
        Configured PlanningMiddleware instance

    Example:
        middleware = create_planning_middleware(auto_plan=True)
        agent = ToolCallAgent(
            middleware=[middleware],
            enable_plan_phase=True,
            enable_reflect_phase=True
        )
    """
    return PlanningMiddleware(
        auto_plan=auto_plan,
        require_explicit_plan=require_explicit_plan
    )
