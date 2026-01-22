"""
Planning Middleware for Deep Agents

Provides automatic planning capabilities for agents:
- Auto-plan generation at the start of tasks
- Plan tracking and execution
- Integration with Plan-Act-Reflect loop

Usage:
    agent = ToolCallAgent(
        middleware=[PlanningMiddleware(auto_plan=True)],
        enable_plan_phase=True,
    )
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Plan Data Structures
# ============================================================================

@dataclass
class PlanStep:
    """A single step in a plan."""
    description: str
    status: str = "pending"  # pending, in_progress, completed, skipped
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def mark_started(self) -> None:
        """Mark step as started."""
        self.status = "in_progress"
        self.started_at = datetime.now()

    def mark_completed(self, result: Optional[str] = None) -> None:
        """Mark step as completed."""
        self.status = "completed"
        self.result = result
        self.completed_at = datetime.now()

    def mark_skipped(self, reason: Optional[str] = None) -> None:
        """Mark step as skipped."""
        self.status = "skipped"
        self.result = reason


@dataclass
class Plan:
    """A plan for completing a task."""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    current_step_index: int = 0

    def add_step(self, description: str) -> PlanStep:
        """Add a step to the plan."""
        step = PlanStep(description=description)
        self.steps.append(step)
        return step

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> bool:
        """Advance to the next step. Returns True if there are more steps."""
        self.current_step_index += 1
        return self.current_step_index < len(self.steps)

    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(s.status in ("completed", "skipped") for s in self.steps)

    def get_progress(self) -> str:
        """Get progress summary."""
        completed = sum(1 for s in self.steps if s.status == "completed")
        total = len(self.steps)
        return f"{completed}/{total} steps completed"

    def to_string(self) -> str:
        """Convert plan to string representation."""
        lines = [f"Goal: {self.goal}", "Steps:"]
        for i, step in enumerate(self.steps):
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "completed": "[x]",
                "skipped": "[-]",
            }.get(step.status, "[ ]")
            marker = "→ " if i == self.current_step_index else "  "
            lines.append(f"{marker}{i+1}. {status_icon} {step.description}")
        return "\n".join(lines)


# ============================================================================
# Planning Middleware
# ============================================================================

class PlanningMiddleware(AgentMiddleware):
    """Middleware that provides automatic planning capabilities.

    This middleware can automatically generate a plan at the start of a task
    and track plan execution progress.

    Features:
    - Auto-plan generation based on task description
    - Plan step tracking
    - Integration with agent's enable_plan_phase

    Usage:
        middleware = PlanningMiddleware(auto_plan=True)

        agent = ToolCallAgent(
            middleware=[middleware],
            enable_plan_phase=True,
        )
    """

    system_prompt = """# Planning Mode

You are operating with planning capabilities. When given a task:
1. First, analyze the task and break it down into clear steps
2. Execute each step methodically
3. Track your progress against the plan
4. Adjust the plan if needed based on new information

Always think about the overall goal and how each action contributes to it.
"""

    def __init__(
        self,
        auto_plan: bool = False,
        max_steps: int = 10,
        plan_prompt: Optional[str] = None,
    ):
        """Initialize planning middleware.

        Args:
            auto_plan: If True, automatically generate a plan at task start
            max_steps: Maximum number of steps in auto-generated plans
            plan_prompt: Custom prompt for plan generation
        """
        super().__init__()
        self.auto_plan = auto_plan
        self.max_steps = max_steps
        self.plan_prompt = plan_prompt or self._default_plan_prompt()

        # Current plan state
        self._current_plan: Optional[Plan] = None
        self._plan_generated: bool = False

        logger.info(f"PlanningMiddleware initialized (auto_plan={auto_plan})")

    def _default_plan_prompt(self) -> str:
        """Get the default planning prompt."""
        return """Analyze this task and create a step-by-step plan to accomplish it.
Break down the task into clear, actionable steps (maximum {max_steps} steps).
Each step should be specific and achievable.

Task: {task}

Provide your plan as a numbered list of steps."""

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime,
    ) -> Optional[Dict[str, Any]]:
        """Initialize planning state before agent runs."""
        # Reset plan state for new run
        self._plan_generated = False

        # Store initial state
        state["_planning_enabled"] = self.auto_plan
        state["_current_plan"] = None

        return state

    def on_plan_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle the plan phase of the agent loop.

        This is called when enable_plan_phase=True on the agent.
        """
        logger.info(f"Planning phase started for agent {runtime.agent_name}")

        # If auto_plan is enabled and we haven't generated a plan yet,
        # the plan will be generated in the next model call
        if self.auto_plan and not self._plan_generated:
            logger.info("Auto-plan enabled, plan will be generated")

        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model calls to inject planning context."""
        # Add current plan context to the request if available
        if self._current_plan:
            plan_context = f"\n\nCurrent Plan:\n{self._current_plan.to_string()}\n"

            # Append to system prompt
            if request.system_prompt:
                request.system_prompt = request.system_prompt + plan_context
            else:
                request.system_prompt = plan_context

        # Call the model
        response = await handler(request)

        # Try to extract plan from response if auto_plan and not yet generated
        if self.auto_plan and not self._plan_generated:
            self._try_extract_plan(response, request)

        return response

    def _try_extract_plan(
        self,
        response: ModelResponse,
        request: ModelRequest,
    ) -> None:
        """Try to extract a plan from the model response."""
        if not response.content:
            return

        # Simple heuristic: look for numbered list patterns
        content = response.content
        lines = content.split("\n")

        steps = []
        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "1)", "- ", "* "
            if (
                (line and line[0].isdigit() and (". " in line[:4] or ") " in line[:4]))
                or line.startswith("- ")
                or line.startswith("* ")
            ):
                # Extract step description
                for sep in [". ", ") ", "- ", "* "]:
                    if sep in line:
                        _, step_desc = line.split(sep, 1)
                        steps.append(step_desc.strip())
                        break

        # Create plan if we found steps
        if steps:
            # Get goal from first user message
            goal = "Complete the task"
            for msg in request.messages:
                if hasattr(msg, 'role') and msg.role == "user":
                    goal = str(msg.content)[:100] if hasattr(msg, 'content') else goal
                    break

            self._current_plan = Plan(goal=goal)
            for step in steps[:self.max_steps]:
                self._current_plan.add_step(step)

            self._plan_generated = True
            logger.info(f"Auto-generated plan with {len(self._current_plan.steps)} steps")

    def on_reflect_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle the reflect phase to update plan progress."""
        if self._current_plan:
            current_step = self._current_plan.get_current_step()
            if current_step and current_step.status == "in_progress":
                # Mark as completed and advance
                current_step.mark_completed()
                self._current_plan.advance()

                logger.info(f"Plan progress: {self._current_plan.get_progress()}")

        return None

    def on_finish_phase(
        self,
        runtime: AgentRuntime,
        phase_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle the finish phase."""
        if self._current_plan:
            logger.info(f"Plan final status: {self._current_plan.get_progress()}")

        return None

    def get_current_plan(self) -> Optional[Plan]:
        """Get the current plan."""
        return self._current_plan

    def set_plan(self, goal: str, steps: List[str]) -> Plan:
        """Manually set a plan.

        Args:
            goal: The goal of the plan
            steps: List of step descriptions

        Returns:
            The created Plan object
        """
        self._current_plan = Plan(goal=goal)
        for step in steps:
            self._current_plan.add_step(step)
        self._plan_generated = True

        logger.info(f"Plan set with {len(steps)} steps")
        return self._current_plan


# ============================================================================
# Convenience Functions
# ============================================================================

def create_planning_middleware(
    auto_plan: bool = True,
    max_steps: int = 10,
) -> PlanningMiddleware:
    """Create a planning middleware with common settings.

    Args:
        auto_plan: Enable automatic plan generation
        max_steps: Maximum steps in auto-generated plans

    Returns:
        Configured PlanningMiddleware
    """
    return PlanningMiddleware(
        auto_plan=auto_plan,
        max_steps=max_steps,
    )
