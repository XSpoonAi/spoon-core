"""
Skill-specific callback handler.

Extends BaseCallbackHandler with skill lifecycle hooks.
"""

import logging
from typing import Any, Dict, Optional

from spoon_ai.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


class SkillCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for skill lifecycle events.

    Extends BaseCallbackHandler with skill-specific hooks:
    - on_skill_start: Called when a skill is activated
    - on_skill_end: Called when a skill is deactivated
    - on_skill_error: Called when skill activation/execution fails

    Usage:
        class MySkillHandler(SkillCallbackHandler):
            async def on_skill_start(self, skill_name, context, **kwargs):
                print(f"Skill activated: {skill_name}")

            async def on_skill_end(self, skill_name, result, **kwargs):
                print(f"Skill deactivated: {skill_name}")

        agent = SpoonReactSkill(callbacks=[MySkillHandler()])
    """

    ignore_skill: bool = False

    async def on_skill_start(
        self,
        skill_name: str,
        context: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Called when a skill is activated.

        Args:
            skill_name: Name of the activated skill
            context: Skill activation context
            run_id: Optional run identifier
            **kwargs: Additional metadata
        """
        pass

    async def on_skill_end(
        self,
        skill_name: str,
        result: Any,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Called when a skill is deactivated.

        Args:
            skill_name: Name of the deactivated skill
            result: Result or state from the skill
            run_id: Optional run identifier
            **kwargs: Additional metadata
        """
        pass

    async def on_skill_error(
        self,
        skill_name: str,
        error: Exception,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Called when a skill encounters an error.

        Args:
            skill_name: Name of the skill that errored
            error: The exception that occurred
            run_id: Optional run identifier
            **kwargs: Additional metadata
        """
        pass

    async def on_skill_match(
        self,
        query: str,
        matched_skills: list,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Called when skills are matched to a query.

        Args:
            query: The user query that triggered matching
            matched_skills: List of skill names that matched
            run_id: Optional run identifier
            **kwargs: Additional metadata
        """
        pass


class LoggingSkillCallback(SkillCallbackHandler):
    """
    Skill callback that logs all events.

    Useful for debugging and monitoring skill system activity.
    """

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level

    async def on_skill_start(
        self,
        skill_name: str,
        context: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        logger.log(
            self.log_level,
            f"[Skill] Activated: {skill_name} | Context: {context} | Run: {run_id}"
        )

    async def on_skill_end(
        self,
        skill_name: str,
        result: Any,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        logger.log(
            self.log_level,
            f"[Skill] Deactivated: {skill_name} | Run: {run_id}"
        )

    async def on_skill_error(
        self,
        skill_name: str,
        error: Exception,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        logger.error(
            f"[Skill] Error in {skill_name}: {error} | Run: {run_id}"
        )

    async def on_skill_match(
        self,
        query: str,
        matched_skills: list,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        logger.log(
            self.log_level,
            f"[Skill] Matched: {matched_skills} for query: {query[:50]}..."
        )


class MetricsSkillCallback(SkillCallbackHandler):
    """
    Skill callback that collects metrics.

    Tracks:
    - Number of skill activations/deactivations
    - Error counts per skill
    - Match counts

    Usage:
        metrics = MetricsSkillCallback()
        agent = SpoonReactSkill(callbacks=[metrics])
        # ... use agent ...
        print(metrics.get_metrics())
    """

    def __init__(self):
        self.activations: Dict[str, int] = {}
        self.deactivations: Dict[str, int] = {}
        self.errors: Dict[str, int] = {}
        self.matches: int = 0
        self.total_matched_skills: int = 0

    async def on_skill_start(
        self,
        skill_name: str,
        context: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        self.activations[skill_name] = self.activations.get(skill_name, 0) + 1

    async def on_skill_end(
        self,
        skill_name: str,
        result: Any,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        self.deactivations[skill_name] = self.deactivations.get(skill_name, 0) + 1

    async def on_skill_error(
        self,
        skill_name: str,
        error: Exception,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        self.errors[skill_name] = self.errors.get(skill_name, 0) + 1

    async def on_skill_match(
        self,
        query: str,
        matched_skills: list,
        *,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        self.matches += 1
        self.total_matched_skills += len(matched_skills)

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "activations": self.activations.copy(),
            "deactivations": self.deactivations.copy(),
            "errors": self.errors.copy(),
            "total_activations": sum(self.activations.values()),
            "total_deactivations": sum(self.deactivations.values()),
            "total_errors": sum(self.errors.values()),
            "match_queries": self.matches,
            "total_matched_skills": self.total_matched_skills
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.activations.clear()
        self.deactivations.clear()
        self.errors.clear()
        self.matches = 0
        self.total_matched_skills = 0
