"""
Skill-enabled production agent.

Combines SpoonReactAI with full skill system support.
"""

import logging
from typing import Optional, List

from pydantic import Field

from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.skill_mixin import SkillEnabledMixin
from spoon_ai.skills.manager import SkillManager

logger = logging.getLogger(__name__)


class SpoonReactSkill(SkillEnabledMixin, SpoonReactAI):
    """
    Production agent with full skill system support.

    Combines:
    - SpoonReactAI: Tool calling, MCP integration, x402 payment
    - SkillEnabledMixin: Skill activation, context injection, auto-trigger

    Usage:
        agent = SpoonReactSkill(name="my_agent")

        # Manual skill activation
        await agent.activate_skill("trading-analysis", {"asset": "BTC"})

        # Run with auto-trigger
        result = await agent.run("Analyze ETH trading signals")
        # -> Automatically activates matching skills

        # List skills
        print(agent.list_skills())
        print(agent.list_active_skills())

        # Deactivate
        await agent.deactivate_skill("trading-analysis")
    """

    name: str = Field(default="spoon_react_skill", description="Agent name")
    description: str = Field(
        default="AI agent with skill system support",
        description="Agent description"
    )

    # Additional skill configuration (mixin provides skill_manager, auto_trigger_skills, max_auto_skills)
    skill_paths: Optional[List[str]] = Field(
        default=None,
        description="Additional paths to search for skills"
    )

    def __init__(self, **kwargs):
        """
        Initialize SpoonReactSkill agent.

        Initializes both SpoonReactAI and skill system components.
        """
        # Extract skill-specific kwargs before parent init
        skill_paths = kwargs.pop('skill_paths', None)

        # Initialize SpoonReactAI
        super().__init__(**kwargs)

        # Store original system prompt before skill injection
        self._original_system_prompt = self.system_prompt
        self._skill_manager_initialized = False

        # Initialize skill manager with agent's LLM
        if self.skill_manager is None:
            llm_manager = None
            if hasattr(self, 'llm') and self.llm:
                try:
                    from spoon_ai.llm.manager import LLMManager
                    if hasattr(self.llm, '_llm_manager'):
                        llm_manager = self.llm._llm_manager
                except ImportError:
                    pass

            self.skill_manager = SkillManager(
                skill_paths=skill_paths,
                llm=llm_manager,
                auto_discover=True
            )

        self._skill_manager_initialized = True

    async def run(self, request: Optional[str] = None) -> str:
        """
        Execute agent with per-turn auto skill activation.

        Flow:
        1. Auto-detect and activate relevant skills (ephemeral for this run)
        2. Sync skill tools into available_tools
        3. Refresh base prompts with current tools
        4. Execute parent SpoonReactAI.run()
        5. Auto-deactivate skills activated in this turn

        Args:
            request: User request/message

        Returns:
            Agent response
        """

        async def _runner(req: Optional[str]) -> str:
            # SpoonReactAI.run() rebuilds prompts from available_tools.
            # Ensure skill tools are synced first, then delegate to parent.
            return await super(SpoonReactSkill, self).run(req)

        return await self._run_with_auto_skills(request, _runner)

    def _map_mcp_tool_name(self, requested_name: str) -> Optional[str]:
        """Map proxy-prefixed MCP tool names to actual server tool names.

        Some OpenAI-compatible gateways may prefix tool names with `proxy_`.
        Keep a local mapping fallback here so MCP calls in ToolCallAgent don't fail
        when this method is absent on skill-enabled agents.
        """
        if not requested_name:
            return None

        # Direct match
        if hasattr(self, "available_tools") and self.available_tools and requested_name in self.available_tools.tool_map:
            return requested_name

        # Strip known proxy prefix
        if requested_name.startswith("proxy_"):
            candidate = requested_name[len("proxy_"):]
            if hasattr(self, "available_tools") and self.available_tools and candidate in self.available_tools.tool_map:
                return candidate
            return candidate

        # No mapping needed / available
        return requested_name

    async def initialize(self, __context=None):
        """
        Initialize async components.

        Extends SpoonReactAI.initialize() to also initialize skill system.
        """
        await super().initialize(__context)

        # Log skill system status
        stats = self.get_skill_stats()
        logger.info(
            f"Skill system initialized: {stats['total_skills']} skills available, "
            f"{len(stats['intent_categories'])} intent categories"
        )

    def add_skill_path(self, path: str) -> None:
        """
        Add a path to search for skills.

        Args:
            path: Directory path to add
        """
        self._ensure_skill_manager().add_skill_path(path)

    def discover_skills(self) -> int:
        """
        Re-discover skills from all configured paths.

        Returns:
            Number of skills discovered
        """
        return self._ensure_skill_manager().discover()
