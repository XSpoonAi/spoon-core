"""
Central skill manager for lifecycle, discovery, and activation.

Reuses:
- IntentAnalyzer from graph/builder.py for LLM-powered matching
- InMemoryCheckpointer from graph/checkpointer.py for state persistence

Script execution support:
- Runs activation/deactivation scripts automatically
- Creates ScriptTool instances for agent access
- Global and per-skill script enable/disable
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

from spoon_ai.skills.models import Skill, SkillState, ScriptResult
from spoon_ai.skills.loader import SkillLoader
from spoon_ai.skills.registry import SkillRegistry
from spoon_ai.skills.executor import get_executor, set_scripts_enabled
from spoon_ai.skills.script_tool import ScriptTool, create_script_tools
from spoon_ai.graph.checkpointer import InMemoryCheckpointer
from spoon_ai.graph.types import StateSnapshot
from spoon_ai.tools.base import BaseTool

if TYPE_CHECKING:
    from spoon_ai.graph.builder import IntentAnalyzer
    from spoon_ai.llm.manager import LLMManager
    from spoon_ai.schema import Message

logger = logging.getLogger(__name__)


# Prompt template for skill intent analysis
SKILL_INTENT_PROMPT = """Analyze the user query and determine which skill category best matches.

Available skill categories:
{categories}

User query: {query}

Respond with a JSON object containing:
- "category": the best matching category name (or "none" if no match)
- "confidence": a number from 0.0 to 1.0 indicating confidence

Example response:
{{"category": "trading_analysis", "confidence": 0.85}}

Your response (JSON only):"""


class SkillManager:
    """
    Central manager for skill lifecycle, discovery, and activation.

    Features:
    - Multi-path skill discovery
    - Keyword and pattern-based trigger matching
    - LLM-powered intent matching (via IntentAnalyzer)
    - State persistence (via InMemoryCheckpointer)
    - Skill composition (prerequisite activation)
    """

    def __init__(
        self,
        skill_paths: Optional[List[str]] = None,
        llm: Optional["LLMManager"] = None,
        auto_discover: bool = True,
        scripts_enabled: bool = True,
        include_default_paths: bool = True
    ):
        """
        Initialize the skill manager.

        Args:
            skill_paths: Additional directories to search for skills
            llm: LLM manager for intent-based matching
            auto_discover: Whether to auto-discover skills on init
            scripts_enabled: Whether to allow script execution globally
        """
        # Convert string paths to Path objects
        additional_paths = None
        if skill_paths:
            additional_paths = [Path(p) for p in skill_paths]

        self._loader = SkillLoader(
            additional_paths=additional_paths,
            include_default_paths=include_default_paths,
        )
        self._registry = SkillRegistry()
        self._checkpointer = InMemoryCheckpointer(
            max_checkpoints_per_thread=10,
            ttl_seconds=3600  # 1 hour
        )

        # Active skills tracking
        self._active_skills: Dict[str, Skill] = {}

        # Script execution tracking
        self._scripts_enabled = scripts_enabled
        self._script_tools: Dict[str, List[ScriptTool]] = {}
        set_scripts_enabled(scripts_enabled)

        # LLM for intent analysis
        self._llm = llm
        self._intent_analyzer: Optional["IntentAnalyzer"] = None

        if auto_discover:
            self.discover()

    def _setup_intent_analyzer(self) -> None:
        """Setup IntentAnalyzer with current skill categories."""
        if not self._llm:
            return

        categories = self._registry.get_intent_categories()
        if not categories:
            logger.debug("No intent categories registered, skipping IntentAnalyzer setup")
            return

        try:
            from spoon_ai.graph.builder import IntentAnalyzer
            from spoon_ai.schema import Message

            def prompt_builder(query: str) -> List["Message"]:
                return [Message(
                    role="user",
                    content=SKILL_INTENT_PROMPT.format(
                        categories=", ".join(categories),
                        query=query
                    )
                )]

            def parser(response: str) -> Dict[str, Any]:
                try:
                    # Try to extract JSON from response
                    response = response.strip()
                    if response.startswith("```"):
                        # Handle markdown code blocks
                        lines = response.split("\n")
                        response = "\n".join(lines[1:-1])
                    return json.loads(response)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse intent response: {response}")
                    return {"category": "none", "confidence": 0.0}

            self._intent_analyzer = IntentAnalyzer(
                llm_manager=self._llm,
                prompt_builder=prompt_builder,
                parser=parser
            )
            logger.debug(f"IntentAnalyzer configured with categories: {categories}")

        except ImportError as e:
            logger.warning(f"Failed to setup IntentAnalyzer: {e}")

    # === Discovery ===

    def discover(self) -> int:
        """
        Discover and register all skills from configured paths.

        Returns:
            Number of skills discovered
        """
        skills = self._loader.load_all()

        for name, skill in skills.items():
            self._registry.register(skill)

        # Setup intent analyzer with discovered categories
        self._setup_intent_analyzer()

        logger.info(f"Discovered {len(skills)} skills")
        return len(skills)

    def add_skill_path(self, path: str) -> None:
        """
        Add a path to search for skills.

        Args:
            path: Directory path to add
        """
        self._loader.add_path(Path(path))

    # === Registry Delegation ===

    def register(self, skill: Skill) -> None:
        """Register a skill manually."""
        self._registry.register(skill)
        self._setup_intent_analyzer()

    def unregister(self, name: str) -> bool:
        """
        Unregister a skill by name.

        Also deactivates if currently active.
        """
        if name in self._active_skills:
            # Synchronous deactivation (state not persisted)
            skill = self._active_skills[name]
            skill.state = SkillState.INACTIVE
            skill.context = {}
            del self._active_skills[name]

        return self._registry.unregister(name)

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._registry.get(name)

    def list(self) -> List[str]:
        """List all registered skill names."""
        return self._registry.list_names()

    def list_skills(self) -> List[Skill]:
        """List all registered skills."""
        return self._registry.list_skills()

    # === Trigger Matching ===

    def match_triggers(self, text: str) -> List[Skill]:
        """
        Match skills by keywords and patterns (fast, no LLM).

        Args:
            text: User input to match against

        Returns:
            List of matching skills, sorted by priority
        """
        return self._registry.find_all_matching(text)

    async def match_intent(self, text: str) -> List[Skill]:
        """
        Match skills by LLM-powered intent analysis.

        Args:
            text: User input to analyze

        Returns:
            List of matching skills
        """
        if not self._intent_analyzer:
            return []

        try:
            intent = await self._intent_analyzer.analyze(text)

            if intent.confidence < 0.5:
                logger.debug(f"Low confidence intent: {intent.category} ({intent.confidence})")
                return []

            if intent.category == "none":
                return []

            return self._registry.find_by_intent(intent.category)

        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            return []

    async def find_matching_skills(self, text: str, use_intent: bool = True) -> List[Skill]:
        """
        Find all matching skills using both trigger and intent matching.

        Args:
            text: User input to match
            use_intent: Whether to also use LLM intent matching

        Returns:
            Combined list of matching skills (deduplicated)
        """
        # Fast trigger matching
        trigger_matches = self.match_triggers(text)

        # LLM intent matching (fallback only when triggers are empty)
        intent_matches = []
        if use_intent and not trigger_matches and self._intent_analyzer:
            intent_matches = await self.match_intent(text)

        # Combine and deduplicate
        seen = set()
        results = []

        for skill in trigger_matches + intent_matches:
            if skill.metadata.name not in seen:
                seen.add(skill.metadata.name)
                results.append(skill)

        return results

    # === Lifecycle ===

    async def activate(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Skill:
        """
        Activate a skill with prerequisite checking.

        Args:
            name: Skill name to activate
            context: Optional context data for the skill

        Returns:
            Activated Skill instance

        Raises:
            ValueError: If skill not found or prerequisites not met
        """
        skill = self._registry.get(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        if name in self._active_skills:
            logger.debug(f"Skill '{name}' already active")
            return self._active_skills[name]

        # Check prerequisites
        await self._check_prerequisites(skill)

        # Activate composed skills first
        for composed_name in skill.metadata.composes:
            if composed_name not in self._active_skills:
                composed_skill = self._registry.get(composed_name)
                if composed_skill and composed_skill.metadata.composable:
                    await self.activate(composed_name)
                else:
                    logger.warning(f"Composed skill '{composed_name}' not found or not composable")

        # Restore persisted state if available
        if skill.metadata.persist_state:
            saved = self._checkpointer.get_checkpoint(f"skill:{name}")
            if saved and saved.values:
                skill.context = saved.values.copy()
                logger.debug(f"Restored state for skill '{name}'")

        # Update skill state
        skill.state = SkillState.ACTIVE
        if context:
            skill.context.update(context)
        self._active_skills[name] = skill

        # Handle script lifecycle
        if self._should_run_scripts(skill):
            # Run activation scripts
            await self._run_activation_scripts(skill)
            # Create script tools for agent access
            self._script_tools[name] = self._create_skill_script_tools(skill)

        logger.info(f"Activated skill: {name}")
        return skill

    async def deactivate(self, name: str) -> bool:
        """
        Deactivate an active skill.

        Persists state if configured. Runs deactivation scripts.

        Args:
            name: Skill name to deactivate

        Returns:
            True if deactivated, False if not active
        """
        if name not in self._active_skills:
            return False

        skill = self._active_skills[name]

        # Run deactivation scripts
        if self._should_run_scripts(skill):
            await self._run_deactivation_scripts(skill)

        # Persist state if configured
        if skill.metadata.persist_state and skill.context:
            snapshot = StateSnapshot(
                values=skill.context.copy(),
                next=set(),
                config={},
                metadata={"skill": name, "deactivated_at": datetime.now().isoformat()},
                created_at=datetime.now()
            )
            self._checkpointer.save_checkpoint(f"skill:{name}", snapshot)
            logger.debug(f"Persisted state for skill '{name}'")

        # Clean up script tools
        if name in self._script_tools:
            del self._script_tools[name]

        # Reset skill state
        skill.state = SkillState.INACTIVE
        skill.context = {}
        del self._active_skills[name]

        logger.info(f"Deactivated skill: {name}")
        return True

    async def deactivate_all(self) -> int:
        """
        Deactivate all active skills.

        Returns:
            Number of skills deactivated
        """
        count = 0
        for name in list(self._active_skills.keys()):
            if await self.deactivate(name):
                count += 1
        return count

    async def _check_prerequisites(self, skill: Skill) -> None:
        """
        Verify skill prerequisites are met.

        Args:
            skill: Skill to check

        Raises:
            ValueError: If prerequisites not met
        """
        prereqs = skill.metadata.prerequisites

        # Check required skills
        for req_skill in prereqs.skills:
            if not self._registry.get(req_skill):
                raise ValueError(
                    f"Skill '{skill.metadata.name}' requires skill '{req_skill}' which is not available"
                )

        # Check required environment variables
        for env_var in prereqs.env_vars:
            if not os.getenv(env_var):
                raise ValueError(
                    f"Skill '{skill.metadata.name}' requires environment variable '{env_var}'"
                )

        # Note: Tool prerequisites could be checked here if ToolManager is available

    # === Context Generation ===

    def get_active_context(self) -> str:
        """
        Generate combined prompt content for all active skills.

        Returns:
            Formatted skill instructions for injection into system prompt
        """
        if not self._active_skills:
            return ""

        sections = ["# Active Skills\n"]

        for name, skill in self._active_skills.items():
            sections.append(skill.get_prompt_injection())
            sections.append("")  # Empty line between skills

        return "\n".join(sections)

    def get_active_tools(self) -> List[BaseTool]:
        """
        Get all tools from active skills.

        Includes both Python tools (from tools.py) and script tools.

        Returns:
            List of tool instances from active skills
        """
        tools = []
        for name in self._active_skills:
            # Python tools from tools.py
            skill_tools = self._loader.get_tools(name)
            tools.extend(skill_tools)

            # Script tools
            if name in self._script_tools:
                tools.extend(self._script_tools[name])

        return tools

    def get_active_skill_names(self) -> List[str]:
        """Get names of all active skills."""
        return list(self._active_skills.keys())

    def is_active(self, name: str) -> bool:
        """Check if a skill is currently active."""
        return name in self._active_skills

    # === Script Execution ===

    def _should_run_scripts(self, skill: Skill) -> bool:
        """
        Check if scripts should be run for a skill.

        Returns True if:
        - Global scripts_enabled is True
        - Skill has scripts defined
        - Skill's scripts are enabled
        """
        if not self._scripts_enabled:
            return False
        if not skill.metadata.has_scripts():
            return False
        if not skill.metadata.scripts_enabled():
            return False
        return True

    def _create_skill_script_tools(self, skill: Skill) -> List[ScriptTool]:
        """
        Create ScriptTool instances for a skill's scripts.

        Args:
            skill: Skill to create tools for

        Returns:
            List of ScriptTool instances
        """
        if not skill.metadata.scripts or not skill.metadata.scripts.definitions:
            return []

        working_dir = skill.metadata.scripts.working_directory
        return create_script_tools(
            skill_name=skill.name,
            scripts=skill.metadata.scripts.definitions,
            working_directory=working_dir
        )

    async def _run_lifecycle_scripts(
        self,
        skill: Skill,
        phase: str
    ) -> List[ScriptResult]:
        """
        Run scripts for a lifecycle phase (activation or deactivation).

        Args:
            skill: Skill being activated/deactivated
            phase: Either "activation" or "deactivation"

        Returns:
            List of execution results
        """
        results = []

        if phase == "activation":
            scripts = skill.metadata.scripts.get_activation_scripts()
        else:
            scripts = skill.metadata.scripts.get_deactivation_scripts()

        if not scripts:
            return results

        executor = get_executor()
        base_working_dir = skill.metadata.scripts.working_directory

        for script in scripts:
            # Per-script working_directory takes precedence over skill-level
            working_dir = script.working_directory or base_working_dir
            logger.info(f"Running {phase} script '{script.name}' for skill '{skill.name}'")
            result = await executor.execute(
                script=script,
                working_directory=working_dir
            )
            results.append(result)

            if not result.success:
                logger.warning(
                    f"{phase.capitalize()} script '{script.name}' failed for skill '{skill.name}': "
                    f"{result.error}"
                )

        return results

    async def _run_activation_scripts(self, skill: Skill) -> List[ScriptResult]:
        """Run scripts marked with run_on_activation."""
        return await self._run_lifecycle_scripts(skill, "activation")

    async def _run_deactivation_scripts(self, skill: Skill) -> List[ScriptResult]:
        """Run scripts marked with run_on_deactivation."""
        return await self._run_lifecycle_scripts(skill, "deactivation")

    async def execute_script(
        self,
        skill_name: str,
        script_name: str,
        input_text: Optional[str] = None
    ) -> ScriptResult:
        """
        Execute a specific script from a skill.

        Args:
            skill_name: Name of the skill containing the script
            script_name: Name of the script to execute
            input_text: Optional input to pass to the script

        Returns:
            ScriptResult with execution details

        Raises:
            ValueError: If skill or script not found
        """
        if not self._scripts_enabled:
            return ScriptResult(
                script_name=script_name,
                success=False,
                error="Script execution is disabled globally"
            )

        skill = self._active_skills.get(skill_name) or self._registry.get(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")

        if not skill.metadata.has_scripts():
            raise ValueError(f"Skill '{skill_name}' has no scripts defined")

        # Check per-skill scripts.enabled flag
        if not skill.metadata.scripts_enabled():
            return ScriptResult(
                script_name=script_name,
                success=False,
                error=f"Script execution is disabled for skill '{skill_name}'"
            )

        script = skill.metadata.scripts.get_script(script_name)
        if not script:
            raise ValueError(f"Script '{script_name}' not found in skill '{skill_name}'")

        executor = get_executor()
        # Per-script working_directory takes precedence over skill-level
        working_dir = script.working_directory or skill.metadata.scripts.working_directory

        return await executor.execute(
            script=script,
            input_text=input_text,
            working_directory=working_dir
        )

    def set_scripts_enabled(self, enabled: bool) -> None:
        """
        Enable or disable script execution globally.

        Args:
            enabled: Whether to enable script execution
        """
        self._scripts_enabled = enabled
        set_scripts_enabled(enabled)
        logger.info(f"Script execution {'enabled' if enabled else 'disabled'}")

    def get_script_tools(self, skill_name: str) -> List[ScriptTool]:
        """
        Get script tools for a specific skill.

        Args:
            skill_name: Name of the skill

        Returns:
            List of ScriptTool instances for the skill
        """
        return self._script_tools.get(skill_name, [])

    # === Info ===

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a skill.

        Args:
            name: Skill name

        Returns:
            Dictionary with skill details or None if not found
        """
        skill = self._registry.get(name)
        if not skill:
            return None

        return {
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "version": skill.metadata.version,
            "author": skill.metadata.author,
            "tags": skill.metadata.tags,
            "state": skill.state.value,
            "is_active": name in self._active_skills,
            "triggers": [
                {
                    "type": t.type,
                    "keywords": t.keywords,
                    "patterns": t.patterns,
                    "intent_category": t.intent_category,
                    "priority": t.priority
                }
                for t in skill.metadata.triggers
            ],
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "default": p.default
                }
                for p in skill.metadata.parameters
            ],
            "prerequisites": {
                "skills": skill.metadata.prerequisites.skills,
                "tools": skill.metadata.prerequisites.tools,
                "env_vars": skill.metadata.prerequisites.env_vars
            },
            "composes": skill.metadata.composes,
            "tool_names": skill.tool_names,
            "script_names": skill.script_names,
            "has_scripts": skill.metadata.has_scripts(),
            "scripts_enabled": skill.metadata.scripts_enabled() if skill.metadata.has_scripts() else False,
            "source_path": skill.source_path
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the skill system."""
        # Get script execution stats
        executor = get_executor()
        script_stats = executor.get_stats()

        return {
            "total_skills": len(self._registry),
            "active_skills": len(self._active_skills),
            "active_skill_names": list(self._active_skills.keys()),
            "skill_paths": [str(p) for p in self._loader.paths],
            "intent_categories": self._registry.get_intent_categories(),
            "has_intent_analyzer": self._intent_analyzer is not None,
            "scripts_enabled": self._scripts_enabled,
            "script_tools_count": sum(len(tools) for tools in self._script_tools.values()),
            "script_execution_stats": script_stats
        }
