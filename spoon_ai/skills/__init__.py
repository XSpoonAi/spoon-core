"""
Skill system for SpoonAI agents.

This module provides a Claude Skills-compatible system for defining and managing
agent capabilities through SKILL.md files and optional Python tools.

Key Components:
- Skill: Data model representing a skill with metadata and triggers
- SkillLoader: Discovers and parses SKILL.md files from configured paths
- SkillRegistry: Thread-safe registry with fast trigger matching
- SkillManager: Central lifecycle manager with LLM intent analysis
- ScriptExecutor: Async subprocess execution for skill scripts

Script Execution:
Skills can define scripts (Python, shell, bash) that agents can execute.
Users control whether scripts are allowed; AI decides how to use them.

Usage:
    from spoon_ai.skills import SkillManager, Skill

    # Create manager with auto-discovery and script support
    manager = SkillManager(auto_discover=True, scripts_enabled=True)

    # Activate a skill
    skill = await manager.activate("research", {"topic": "AI"})

    # Get prompt injection for active skills
    context = manager.get_active_context()

    # Execute a skill script
    result = await manager.execute_script("data-processor", "analyze")

    # Find matching skills for user input
    matches = await manager.find_matching_skills("research quantum computing")
"""

from spoon_ai.skills.models import (
    Skill,
    SkillState,
    SkillMetadata,
    SkillTrigger,
    SkillParameter,
    SkillPrerequisite,
    # Script models
    ScriptType,
    SkillScript,
    ScriptConfig,
    ScriptResult,
)
from spoon_ai.skills.loader import SkillLoader
from spoon_ai.skills.registry import SkillRegistry
from spoon_ai.skills.manager import SkillManager
from spoon_ai.skills.executor import (
    ScriptExecutor,
    ScriptExecutionError,
    get_executor,
    configure_executor,
    set_scripts_enabled,
)
from spoon_ai.skills.script_tool import ScriptTool, create_script_tools

__all__ = [
    # Models
    "Skill",
    "SkillState",
    "SkillMetadata",
    "SkillTrigger",
    "SkillParameter",
    "SkillPrerequisite",
    # Script models
    "ScriptType",
    "SkillScript",
    "ScriptConfig",
    "ScriptResult",
    # Components
    "SkillLoader",
    "SkillRegistry",
    "SkillManager",
    # Script execution
    "ScriptExecutor",
    "ScriptExecutionError",
    "get_executor",
    "configure_executor",
    "set_scripts_enabled",
    "ScriptTool",
    "create_script_tools",
]
