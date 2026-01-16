"""
Skill-enabled agent mixin.

Follows MCPClientMixin pattern for composable agent integration.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import Field

from spoon_ai.skills.manager import SkillManager
from spoon_ai.skills.models import Skill

if TYPE_CHECKING:
    from spoon_ai.tools import ToolManager

logger = logging.getLogger(__name__)


class SkillEnabledMixin:
    """
    Mixin that adds skill capabilities to agents.

    Integrates with ReAct cycle by:
    1. Injecting active skill instructions into system prompt
    2. Adding skill tools to available_tools
    3. Auto-triggering skills based on user input

    Usage:
        class MyAgent(SkillEnabledMixin, SpoonReactAI):
            pass

        agent = MyAgent()
        await agent.activate_skill("trading-analysis")
        result = await agent.run("Analyze BTC")
    """

    # Pydantic fields (when used with BaseModel agents)
    skill_manager: Optional[SkillManager] = Field(
        default=None,
        description="Skill manager instance"
    )
    auto_trigger_skills: bool = Field(
        default=True,
        description="Whether to auto-activate skills based on user input"
    )
    max_auto_skills: int = Field(
        default=3,
        description="Maximum number of skills to auto-activate"
    )

    # Private state
    _original_system_prompt: Optional[str] = None
    _skill_manager_initialized: bool = False

    def _ensure_skill_manager(self) -> SkillManager:
        """
        Ensure skill manager is initialized.

        Lazily creates SkillManager with agent's LLM if available.
        """
        if self.skill_manager is None:
            # Try to get LLM from agent
            llm = getattr(self, 'llm', None)
            llm_manager = None

            if llm:
                # Try to get LLMManager from ChatBot
                try:
                    from spoon_ai.llm.manager import LLMManager
                    if hasattr(llm, '_llm_manager'):
                        llm_manager = llm._llm_manager
                    elif isinstance(llm, LLMManager):
                        llm_manager = llm
                except ImportError:
                    pass

            self.skill_manager = SkillManager(llm=llm_manager)

        if not self._skill_manager_initialized:
            self._original_system_prompt = getattr(self, 'system_prompt', None)
            self._skill_manager_initialized = True

        return self.skill_manager

    def _refresh_prompts_with_skills(self) -> None:
        """
        Inject active skill instructions into system prompt.

        Follows SpoonReactAI._refresh_prompts() pattern.
        """
        manager = self._ensure_skill_manager()
        base_prompt = self._original_system_prompt or ""
        skill_context = manager.get_active_context()

        if skill_context:
            self.system_prompt = f"{base_prompt}\n\n{skill_context}"
        else:
            self.system_prompt = base_prompt

    def _inject_skill_tools(self) -> None:
        """
        Add tools from active skills to available_tools.

        Only adds tools not already present.
        """
        available_tools: Optional["ToolManager"] = getattr(self, 'available_tools', None)
        if not available_tools:
            return

        manager = self._ensure_skill_manager()
        skill_tools = manager.get_active_tools()

        for tool in skill_tools:
            if tool.name not in available_tools.tool_map:
                available_tools.add_tool(tool)
                logger.debug(f"Injected skill tool: {tool.name}")

    async def activate_skill(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Skill:
        """
        Activate a skill and refresh agent state.

        Args:
            name: Skill name to activate
            context: Optional context data

        Returns:
            Activated Skill instance
        """
        manager = self._ensure_skill_manager()
        skill = await manager.activate(name, context)

        # Refresh agent state
        self._refresh_prompts_with_skills()
        self._inject_skill_tools()

        agent_name = getattr(self, 'name', 'unknown')
        logger.info(f"Agent '{agent_name}' activated skill: {name}")

        return skill

    async def deactivate_skill(self, name: str) -> bool:
        """
        Deactivate a skill.

        Args:
            name: Skill name to deactivate

        Returns:
            True if deactivated, False if not active
        """
        manager = self._ensure_skill_manager()
        result = await manager.deactivate(name)

        if result:
            self._refresh_prompts_with_skills()
            # Note: We don't remove tools as they might still be useful

        return result

    async def auto_activate_skills(self, user_input: str) -> List[Skill]:
        """
        Automatically activate skills matching user input.

        Uses both keyword/pattern matching and LLM intent analysis.

        Args:
            user_input: User's message

        Returns:
            List of activated skills
        """
        if not self.auto_trigger_skills:
            return []

        manager = self._ensure_skill_manager()

        # Find matching skills
        matches = await manager.find_matching_skills(user_input, use_intent=True)

        if not matches:
            return []

        # Activate top matches (limited by max_auto_skills)
        activated = []
        for skill in matches[:self.max_auto_skills]:
            # Skip already active skills
            if manager.is_active(skill.metadata.name):
                continue

            try:
                await self.activate_skill(skill.metadata.name)
                activated.append(skill)
            except Exception as e:
                logger.warning(f"Failed to auto-activate skill '{skill.metadata.name}': {e}")

        if activated:
            names = [s.metadata.name for s in activated]
            logger.info(f"Auto-activated skills: {names}")

        return activated

    def list_skills(self) -> List[str]:
        """List all available skill names."""
        manager = self._ensure_skill_manager()
        return manager.list()

    def list_active_skills(self) -> List[str]:
        """List currently active skill names."""
        manager = self._ensure_skill_manager()
        return manager.get_active_skill_names()

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a skill."""
        manager = self._ensure_skill_manager()
        return manager.get_skill_info(name)

    def is_skill_active(self, name: str) -> bool:
        """Check if a skill is currently active."""
        manager = self._ensure_skill_manager()
        return manager.is_active(name)

    async def deactivate_all_skills(self) -> int:
        """
        Deactivate all active skills.

        Returns:
            Number of skills deactivated
        """
        manager = self._ensure_skill_manager()
        count = await manager.deactivate_all()

        if count > 0:
            self._refresh_prompts_with_skills()

        return count

    def get_skill_stats(self) -> Dict[str, Any]:
        """Get skill system statistics."""
        manager = self._ensure_skill_manager()
        return manager.get_stats()
