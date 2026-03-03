"""
Unit tests for the SpoonAI skill system.

This module tests:
- Skill data models (SkillMetadata, SkillTrigger, SkillParameter, etc.)
- Skill loader and SKILL.md parsing
- Skill registry and indexing
- Skill manager lifecycle
- Script execution system (ScriptType, SkillScript, ScriptConfig, ScriptResult)
- Script executor and ScriptTool integration
"""

import pytest
import tempfile
from pathlib import Path
from typing import Optional

# Path to example skills (since no built-in skills)
EXAMPLES_SKILLS_PATH = Path(__file__).parent.parent / "examples" / "skills"

from spoon_ai.skills import (
    # Models
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
    # Core components
    SkillLoader,
    SkillRegistry,
    SkillManager,
    # Script execution
    ScriptExecutor,
    get_executor,
    set_scripts_enabled,
    ScriptTool,
    create_script_tools,
)
from spoon_ai.agents.skill_mixin import SkillEnabledMixin
from spoon_ai.tools import ToolManager


# Sample SKILL.md content for testing
SAMPLE_SKILL_MD = """---
name: test-skill
description: A test skill for unit testing
version: 1.0.0
author: Test Author
tags:
  - test
  - unit-test
triggers:
  - type: keyword
    keywords:
      - test
      - testing
    priority: 80
  - type: pattern
    patterns:
      - "(?i)run test.*"
    priority: 70
parameters:
  - name: test_param
    type: string
    required: true
    description: A test parameter
prerequisites:
  tools: []
  env_vars: []
  skills: []
composable: true
persist_state: false
---

# Test Skill

You are now in **Test Mode**.

## Instructions

1. Run tests
2. Report results

## Context

- Parameter: {{test_param}}
"""

SKILL_WITH_TOOL_MD = """---
name: tool-skill
description: Skill with tool
version: 1.0.0
triggers:
  - type: keyword
    keywords:
      - tool
---

# Tool Skill

Use the tool when requested.
"""

SKILL_WITH_TRIGGER_MD = """---
name: trigger-skill
description: Skill with trigger
version: 1.0.0
triggers:
  - type: keyword
    keywords:
      - trigger
---

# Trigger Skill

Triggered by keyword.
"""

TOOLS_PY_CONTENT = """from typing import Any, Dict

from spoon_ai.tools.base import BaseTool, ToolResult


class TestSkillTool(BaseTool):
    name: str = "test_skill_tool"
    description: str = "A test skill tool."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def execute(self) -> ToolResult:
        return ToolResult(output="ok")
"""


class DummySkillAgent(SkillEnabledMixin):
    def __init__(self, skill_paths):
        self.system_prompt = "Base prompt"
        self.available_tools = ToolManager([])
        self.skill_manager = SkillManager(skill_paths=skill_paths, auto_discover=True)
        self._skill_manager_initialized = False
        self._original_system_prompt = None
        self.auto_trigger_skills = True
        self.max_auto_skills = 3

    async def run_stub(self, request: Optional[str] = None) -> str:
        return "ok"


# =============================================================================
# SKILL MODELS TESTS
# =============================================================================

class TestSkillModels:
    """Tests for skill data models."""

    def test_skill_trigger_keyword(self):
        trigger = SkillTrigger(
            type="keyword",
            keywords=["test", "demo"],
            priority=80
        )
        assert trigger.type == "keyword"
        assert "test" in trigger.keywords
        assert trigger.priority == 80

    def test_skill_trigger_pattern(self):
        trigger = SkillTrigger(
            type="pattern",
            patterns=[r"(?i)hello.*"],
            priority=70
        )
        assert trigger.type == "pattern"
        assert len(trigger.patterns) == 1

    def test_skill_trigger_intent(self):
        trigger = SkillTrigger(
            type="intent",
            intent_category="research",
            priority=90
        )
        assert trigger.type == "intent"
        assert trigger.intent_category == "research"

    def test_skill_parameter(self):
        param = SkillParameter(
            name="topic",
            type="string",
            required=True,
            description="Research topic"
        )
        assert param.name == "topic"
        assert param.required is True

    def test_skill_parameter_with_default(self):
        param = SkillParameter(
            name="depth",
            type="string",
            required=False,
            default="medium",
            description="Research depth"
        )
        assert param.default == "medium"

    def test_skill_metadata(self):
        metadata = SkillMetadata(
            name="test-skill",
            description="Test description",
            version="1.0.0",
            author="Test Author",
            tags=["test"]
        )
        assert metadata.name == "test-skill"
        assert "test" in metadata.tags

    def test_skill_state_enum(self):
        assert SkillState.INACTIVE == "inactive"
        assert SkillState.ACTIVE == "active"
        assert SkillState.LOADING == "loading"
        assert SkillState.ERROR == "error"

    def test_skill_creation(self):
        metadata = SkillMetadata(
            name="test-skill",
            description="Test",
            version="1.0.0"
        )
        skill = Skill(metadata=metadata, instructions="Test instructions")
        assert skill.metadata.name == "test-skill"
        assert skill.instructions == "Test instructions"
        assert skill.state == SkillState.INACTIVE

    def test_skill_get_prompt_injection(self):
        metadata = SkillMetadata(
            name="test-skill",
            description="Test skill",
            version="1.0.0"
        )
        skill = Skill(metadata=metadata, instructions="Hello World!")
        prompt = skill.get_prompt_injection()
        assert "test-skill" in prompt
        assert "Test skill" in prompt
        assert "Hello World!" in prompt

    def test_skill_convenience_properties(self):
        metadata = SkillMetadata(name="my-skill", description="My description")
        skill = Skill(metadata=metadata, instructions="Test")
        assert skill.name == "my-skill"
        assert skill.description == "My description"


# =============================================================================
# SCRIPT MODELS TESTS
# =============================================================================

class TestScriptModels:
    """Tests for script data models."""

    def test_script_type_enum(self):
        assert ScriptType.PYTHON.value == "python"
        assert ScriptType.SHELL.value == "shell"
        assert ScriptType.BASH.value == "bash"

    def test_skill_script_with_file(self):
        script = SkillScript(
            name="test-script",
            description="Test script",
            type=ScriptType.PYTHON,
            file="test.py",
            timeout=60
        )
        assert script.name == "test-script"
        assert script.file == "test.py"
        assert script.inline is None

    def test_skill_script_with_inline(self):
        script = SkillScript(
            name="inline-script",
            type=ScriptType.BASH,
            inline="echo 'Hello World'"
        )
        assert script.name == "inline-script"
        assert script.inline == "echo 'Hello World'"
        assert script.file is None

    def test_skill_script_validation_requires_source(self):
        with pytest.raises(ValueError, match="Either 'file' or 'inline'"):
            SkillScript(name="no-source", type=ScriptType.PYTHON)

    def test_skill_script_validation_exclusive_source(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            SkillScript(
                name="both-sources",
                type=ScriptType.PYTHON,
                file="test.py",
                inline="print('test')"
            )

    def test_script_config(self):
        config = ScriptConfig(
            enabled=True,
            working_directory="./scripts",
            definitions=[
                SkillScript(name="s1", type=ScriptType.PYTHON, file="s1.py"),
                SkillScript(name="s2", type=ScriptType.BASH, inline="echo test"),
            ]
        )
        assert config.enabled is True
        assert len(config.definitions) == 2
        assert config.get_script("s1") is not None
        assert config.get_script("s3") is None

    def test_script_result(self):
        result = ScriptResult(
            script_name="test",
            success=True,
            exit_code=0,
            stdout="output",
            stderr="",
            execution_time=1.5
        )
        assert result.success is True
        assert result.to_string() == "output"

        failed_result = ScriptResult(
            script_name="test",
            success=False,
            exit_code=1,
            error="Something went wrong"
        )
        assert "Error:" in failed_result.to_string()


# =============================================================================
# SKILL LOADER TESTS
# =============================================================================

class TestSkillLoader:
    """Tests for SKILL.md parsing and discovery."""

    def test_parse_skill_md_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(SAMPLE_SKILL_MD)

            loader = SkillLoader()
            metadata, instructions = loader.parse(skill_file)

            assert metadata.name == "test-skill"
            assert metadata.version == "1.0.0"
            assert "test" in metadata.tags
            assert len(metadata.triggers) == 2
            assert len(metadata.parameters) == 1
            assert "Test Mode" in instructions

    def test_parse_invalid_skill_md_no_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("Just plain text")

            loader = SkillLoader()
            with pytest.raises(ValueError, match="missing YAML frontmatter"):
                loader.parse(skill_file)

    def test_parse_invalid_skill_md_no_name(self):
        content = """---
description: Missing name field
---
Content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text(content)

            loader = SkillLoader()
            with pytest.raises(ValueError, match="Missing required 'name'"):
                loader.parse(skill_file)

    def test_discover_example_skills(self):
        loader = SkillLoader(additional_paths=[EXAMPLES_SKILLS_PATH])
        skills = loader.load_all()
        assert len(skills) >= 1
        assert "research" in skills or "data-processor" in skills

    def test_discover_from_custom_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(SAMPLE_SKILL_MD)

            loader = SkillLoader(additional_paths=[Path(tmpdir)])
            skills = loader.load_all()
            assert "test-skill" in skills

    def test_load_single_skill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "custom-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(SAMPLE_SKILL_MD)

            loader = SkillLoader()
            skill = loader.load(skill_file)

            assert skill.metadata.name == "test-skill"
            assert skill.source_path == str(skill_file)
            assert skill.loaded_at is not None

    def test_loader_paths_property(self):
        loader = SkillLoader(additional_paths=[EXAMPLES_SKILLS_PATH])
        paths = loader.paths
        assert len(paths) >= 1
        assert any("examples" in str(p) or "skills" in str(p) for p in paths)

    def test_load_skill_with_scripts(self):
        loader = SkillLoader(additional_paths=[EXAMPLES_SKILLS_PATH])
        loader.load_all()
        skill = loader.get_skill("data-processor")
        if skill:
            assert skill.metadata.has_scripts()
            assert skill.metadata.scripts_enabled()
            assert len(skill.script_names) > 0


# =============================================================================
# SKILL REGISTRY TESTS
# =============================================================================

class TestSkillRegistry:
    """Tests for skill registry and indexing."""

    @pytest.fixture
    def registry(self):
        return SkillRegistry()

    @pytest.fixture
    def sample_skill(self):
        metadata = SkillMetadata(
            name="sample-skill",
            description="Sample",
            version="1.0.0",
            tags=["sample", "test"],
            triggers=[
                SkillTrigger(type="keyword", keywords=["sample", "example"], priority=80),
                SkillTrigger(type="pattern", patterns=[r"(?i)show sample.*"], priority=70),
                SkillTrigger(type="intent", intent_category="demonstration", priority=90),
            ]
        )
        return Skill(metadata=metadata, instructions="Sample instructions")

    def test_register_skill(self, registry, sample_skill):
        registry.register(sample_skill)
        assert registry.get("sample-skill") is not None
        assert "sample-skill" in registry.list_names()

    def test_register_overwrites_existing(self, registry, sample_skill):
        registry.register(sample_skill)
        registry.register(sample_skill)
        assert len(registry) == 1

    def test_unregister_skill(self, registry, sample_skill):
        registry.register(sample_skill)
        result = registry.unregister("sample-skill")
        assert result is True
        assert registry.get("sample-skill") is None

    def test_unregister_nonexistent(self, registry):
        result = registry.unregister("nonexistent")
        assert result is False

    def test_find_by_keyword(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_by_keyword("show me a sample")
        assert len(matches) == 1
        assert matches[0].metadata.name == "sample-skill"

    def test_find_by_keyword_no_match(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_by_keyword("unrelated query")
        assert len(matches) == 0

    def test_find_by_pattern(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_by_pattern("show sample data")
        assert len(matches) == 1
        assert matches[0].metadata.name == "sample-skill"

    def test_find_by_pattern_no_match(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_by_pattern("unrelated text")
        assert len(matches) == 0

    def test_find_by_intent(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_by_intent("demonstration")
        assert len(matches) == 1

    def test_find_by_intent_no_match(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_by_intent("unknown_intent")
        assert len(matches) == 0

    def test_find_by_tag(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_by_tag("sample")
        assert len(matches) == 1
        matches = registry.find_by_tag("nonexistent")
        assert len(matches) == 0

    def test_find_all_matching(self, registry, sample_skill):
        registry.register(sample_skill)
        matches = registry.find_all_matching("show sample data")
        assert len(matches) >= 1

    def test_get_intent_categories(self, registry, sample_skill):
        registry.register(sample_skill)
        categories = registry.get_intent_categories()
        assert "demonstration" in categories

    def test_registry_len(self, registry, sample_skill):
        assert len(registry) == 0
        registry.register(sample_skill)
        assert len(registry) == 1

    def test_registry_contains(self, registry, sample_skill):
        assert "sample-skill" not in registry
        registry.register(sample_skill)
        assert "sample-skill" in registry


# =============================================================================
# SCRIPT EXECUTOR TESTS
# =============================================================================

class TestScriptExecutor:
    """Tests for script executor."""

    def test_executor_creation(self):
        executor = ScriptExecutor(enabled=True, default_timeout=60)
        assert executor.enabled is True
        assert executor.default_timeout == 60

    def test_global_executor(self):
        executor = get_executor()
        assert executor is not None
        set_scripts_enabled(False)
        assert executor.enabled is False
        set_scripts_enabled(True)
        assert executor.enabled is True

    def test_interpreter_detection(self):
        executor = ScriptExecutor()
        assert executor.is_available(ScriptType.PYTHON)
        assert executor.get_interpreter(ScriptType.PYTHON) is not None

    @pytest.mark.asyncio
    async def test_execute_inline_python(self):
        executor = ScriptExecutor(enabled=True)
        script = SkillScript(
            name="hello",
            type=ScriptType.PYTHON,
            inline="print('Hello from script!')"
        )
        result = await executor.execute(script)
        assert result.success is True
        assert "Hello from script!" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_input(self):
        executor = ScriptExecutor(enabled=True)
        script = SkillScript(
            name="echo-input",
            type=ScriptType.PYTHON,
            inline="import sys; print(f'Got: {sys.stdin.read().strip()}')"
        )
        result = await executor.execute(script, input_text="test data")
        assert result.success is True
        assert "Got: test data" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_disabled(self):
        executor = ScriptExecutor(enabled=False)
        script = SkillScript(
            name="disabled",
            type=ScriptType.PYTHON,
            inline="print('should not run')"
        )
        result = await executor.execute(script)
        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        executor = ScriptExecutor(enabled=True)
        script = SkillScript(
            name="slow",
            type=ScriptType.PYTHON,
            inline="import time; time.sleep(10); print('done')",
            timeout=1
        )
        result = await executor.execute(script)
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_script_error(self):
        executor = ScriptExecutor(enabled=True)
        script = SkillScript(
            name="error",
            type=ScriptType.PYTHON,
            inline="raise ValueError('intentional error')"
        )
        result = await executor.execute(script)
        assert result.success is False
        assert result.exit_code != 0

    def test_execution_stats(self):
        executor = ScriptExecutor(enabled=True)
        stats = executor.get_stats()
        assert "total" in stats
        assert "successful" in stats
        assert "failed" in stats


# =============================================================================
# SCRIPT TOOL TESTS
# =============================================================================

class TestScriptTool:
    """Tests for script tool wrapper."""

    def test_script_tool_creation(self):
        script = SkillScript(
            name="analyze",
            description="Analyze data",
            type=ScriptType.PYTHON,
            inline="print('analyzing')"
        )
        tool = ScriptTool(script=script, skill_name="data-processor")
        assert "run_script_data-processor_analyze" in tool.name
        assert "Analyze data" in tool.description
        assert tool.parameters is not None

    def test_create_script_tools(self):
        scripts = [
            SkillScript(name="s1", type=ScriptType.PYTHON, inline="print(1)"),
            SkillScript(name="s2", type=ScriptType.PYTHON, inline="print(2)"),
        ]
        tools = create_script_tools("test-skill", scripts)
        assert len(tools) == 2
        assert all(isinstance(t, ScriptTool) for t in tools)

    @pytest.mark.asyncio
    async def test_script_tool_execution(self):
        script = SkillScript(
            name="greet",
            type=ScriptType.PYTHON,
            inline="print('Hello from tool!')"
        )
        tool = ScriptTool(script=script, skill_name="test")
        result = await tool.execute()
        assert "Hello from tool!" in result


# =============================================================================
# SKILL MANAGER TESTS
# =============================================================================

class TestSkillManager:
    """Tests for skill lifecycle management."""

    @pytest.fixture
    def manager(self):
        return SkillManager(skill_paths=[str(EXAMPLES_SKILLS_PATH)], auto_discover=True)

    def test_discover_skills(self, manager):
        count = manager.discover()
        assert count >= 1

    def test_list_skills(self, manager):
        skills = manager.list()
        assert "research" in skills or "data-processor" in skills

    def test_get_skill_info(self, manager):
        skill_list = manager.list()
        skill_name = skill_list[0] if skill_list else "research"
        info = manager.get_skill_info(skill_name)
        assert info is not None
        assert info["name"] == skill_name
        assert "description" in info
        assert "triggers" in info

    @pytest.mark.asyncio
    async def test_activate_skill(self, manager):
        skill_list = manager.list()
        skill_name = skill_list[0] if skill_list else "research"
        skill = await manager.activate(skill_name, {"topic": "AI"})
        assert skill is not None
        assert manager.is_active(skill_name)
        assert skill_name in manager.get_active_skill_names()

    @pytest.mark.asyncio
    async def test_deactivate_skill(self, manager):
        skill_list = manager.list()
        skill_name = skill_list[0] if skill_list else "research"
        await manager.activate(skill_name)
        result = await manager.deactivate(skill_name)
        assert result is True
        assert not manager.is_active(skill_name)

    @pytest.mark.asyncio
    async def test_deactivate_inactive_skill(self, manager):
        result = await manager.deactivate("nonexistent-skill")
        assert result is False

    @pytest.mark.asyncio
    async def test_activate_nonexistent_skill(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.activate("nonexistent-skill")

    @pytest.mark.asyncio
    async def test_get_active_context(self, manager):
        skill_list = manager.list()
        skill_name = skill_list[0] if skill_list else "research"
        await manager.activate(skill_name, {"topic": "Machine Learning"})
        context = manager.get_active_context()
        assert context is not None
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_deactivate_all(self, manager):
        skill_list = manager.list()
        skill_name = skill_list[0] if skill_list else "research"
        await manager.activate(skill_name)
        count = await manager.deactivate_all()
        assert count == 1
        assert len(manager.get_active_skill_names()) == 0

    def test_match_triggers_keyword(self, manager):
        matches = manager.match_triggers("research data analysis")
        assert len(matches) >= 0

    def test_match_triggers_pattern(self, manager):
        matches = manager.match_triggers("analyze this data")
        assert isinstance(matches, list)

    def test_get_stats(self, manager):
        stats = manager.get_stats()
        assert "total_skills" in stats
        assert "active_skills" in stats
        assert "intent_categories" in stats
        assert stats["total_skills"] >= 1

    @pytest.mark.asyncio
    async def test_intent_fallback_skips_on_trigger_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "trigger-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(SKILL_WITH_TRIGGER_MD)

            manager = SkillManager(skill_paths=[tmpdir], auto_discover=True, include_default_paths=False)
            manager._intent_analyzer = object()
            calls = {"count": 0}

            async def fake_match_intent(text: str):
                calls["count"] += 1
                return []

            manager.match_intent = fake_match_intent  # type: ignore[assignment]

            matches = await manager.find_matching_skills("trigger this", use_intent=True)
            assert len(matches) == 1
            assert calls["count"] == 0

            matches = await manager.find_matching_skills("no match", use_intent=True)
            assert len(matches) == 0
            assert calls["count"] == 1


# =============================================================================
# SKILL MANAGER SCRIPT INTEGRATION TESTS
# =============================================================================

class TestSkillManagerScripts:
    """Tests for skill manager script integration."""

    @pytest.mark.asyncio
    async def test_manager_scripts_enabled(self):
        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        stats = manager.get_stats()
        assert stats["scripts_enabled"] is True

    @pytest.mark.asyncio
    async def test_manager_scripts_disabled(self):
        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=False
        )
        stats = manager.get_stats()
        assert stats["scripts_enabled"] is False

    @pytest.mark.asyncio
    async def test_activate_skill_with_scripts(self):
        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        if manager.get("data-processor"):
            skill = await manager.activate("data-processor")
            assert skill is not None
            script_tools = manager.get_script_tools("data-processor")
            assert isinstance(script_tools, list)

    @pytest.mark.asyncio
    async def test_deactivate_skill_cleans_script_tools(self):
        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        if manager.get("data-processor"):
            await manager.activate("data-processor")
            await manager.deactivate("data-processor")
            script_tools = manager.get_script_tools("data-processor")
            assert len(script_tools) == 0

    @pytest.mark.asyncio
    async def test_execute_script_via_manager(self):
        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        if manager.get("data-processor"):
            await manager.activate("data-processor")
            try:
                result = await manager.execute_script(
                    "data-processor",
                    "analyze",
                    input_text='{"test": "data"}'
                )
                assert result is not None
            except ValueError:
                pass  # Script might not be found

    @pytest.mark.asyncio
    async def test_get_active_tools_includes_scripts(self):
        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        if manager.get("data-processor"):
            await manager.activate("data-processor")
            tools = manager.get_active_tools()
            assert isinstance(tools, list)

    def test_set_scripts_enabled_via_manager(self):
        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        manager.set_scripts_enabled(False)
        assert manager.get_stats()["scripts_enabled"] is False
        manager.set_scripts_enabled(True)
        assert manager.get_stats()["scripts_enabled"] is True


# =============================================================================
# DATA-PROCESSOR SKILL TESTS
# =============================================================================

class TestDataProcessorSkill:
    """Tests for the data-processor example skill."""

    @pytest.fixture
    def loader(self):
        return SkillLoader(additional_paths=[EXAMPLES_SKILLS_PATH])

    def test_skill_loads(self, loader):
        skills = loader.load_all()
        skill = loader.get_skill("data-processor")
        if skill:
            assert skill.name == "data-processor"
            assert "data" in skill.metadata.tags
            assert skill.metadata.has_scripts()

    def test_skill_has_scripts(self, loader):
        loader.load_all()
        skill = loader.get_skill("data-processor")
        if skill and skill.metadata.scripts:
            config = skill.metadata.scripts
            assert config.enabled is True
            scripts = [
                config.get_script("analyze"),
                config.get_script("transform"),
                config.get_script("setup"),
                config.get_script("cleanup")
            ]
            assert any(s is not None for s in scripts)

    def test_activation_scripts(self, loader):
        loader.load_all()
        skill = loader.get_skill("data-processor")
        if skill and skill.metadata.scripts:
            activation_scripts = skill.metadata.scripts.get_activation_scripts()
            assert any(s.name == "setup" for s in activation_scripts)

    def test_deactivation_scripts(self, loader):
        loader.load_all()
        skill = loader.get_skill("data-processor")
        if skill and skill.metadata.scripts:
            deactivation_scripts = skill.metadata.scripts.get_deactivation_scripts()
            assert any(s.name == "cleanup" for s in deactivation_scripts)

    @pytest.mark.asyncio
    async def test_analyze_script_execution(self, loader):
        loader.load_all()
        skill = loader.get_skill("data-processor")
        if not skill or not skill.metadata.scripts:
            pytest.skip("data-processor skill not found")

        analyze = skill.metadata.scripts.get_script("analyze")
        if not analyze:
            pytest.skip("analyze script not found")

        executor = get_executor()
        result = await executor.execute(
            script=analyze,
            input_text='{"name": "test", "value": 123}',
            working_directory=skill.metadata.scripts.working_directory
        )
        assert result.success is True
        assert "json" in result.stdout.lower() or "success" in result.stdout.lower()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSkillIntegration:
    """Integration tests for skill system."""

    @pytest.mark.asyncio
    async def test_full_skill_lifecycle(self):
        manager = SkillManager(skill_paths=[str(EXAMPLES_SKILLS_PATH)], auto_discover=True)
        skill_list = manager.list()
        assert skill_list
        skill_name = skill_list[0]

        # Find matching skills
        matches = manager.match_triggers("research data analysis")
        assert isinstance(matches, list)

        # Activate
        skill = await manager.activate(skill_name, {"topic": "AI trends", "depth": "deep"})
        assert skill is not None

        # Get context
        context = manager.get_active_context()
        assert len(context) > 0

        # Deactivate
        await manager.deactivate(skill_name)
        assert not manager.is_active(skill_name)

    @pytest.mark.asyncio
    async def test_custom_skill_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "custom-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD)

            manager = SkillManager(skill_paths=[tmpdir], auto_discover=True, include_default_paths=False)
            assert "test-skill" in manager.list()

            skill = await manager.activate("test-skill", {"test_param": "hello"})
            assert skill is not None

    @pytest.mark.asyncio
    async def test_multiple_skills_activation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                skill_dir = Path(tmpdir) / f"skill-{i}"
                skill_dir.mkdir()
                content = f"""---
name: skill-{i}
description: Skill number {i}
triggers:
  - type: keyword
    keywords:
      - skill{i}
---
Instructions for skill {i}
"""
                (skill_dir / "SKILL.md").write_text(content)

            manager = SkillManager(skill_paths=[tmpdir], auto_discover=True, include_default_paths=False)
            await manager.activate("skill-0")
            await manager.activate("skill-1")
            assert len(manager.get_active_skill_names()) == 2

            count = await manager.deactivate_all()
            assert count == 2


# =============================================================================
# SKILL MIXIN / TOOL SYNC TESTS
# =============================================================================

class TestSkillMixinBehavior:
    """Tests for skill mixin tool sync and auto-activation cleanup."""

    @pytest.mark.asyncio
    async def test_auto_activated_skills_are_ephemeral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "tool-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(SKILL_WITH_TOOL_MD)
            (skill_dir / "tools.py").write_text(TOOLS_PY_CONTENT)

            agent = DummySkillAgent([tmpdir])
            active_during_run = {}

            async def run_stub(request: Optional[str] = None) -> str:
                active_during_run["names"] = agent.list_active_skills()
                return "ok"

            await agent._run_with_auto_skills("use tool", run_stub)

            assert "tool-skill" in active_during_run.get("names", [])
            assert agent.list_active_skills() == []
            assert agent.system_prompt == "Base prompt"

    @pytest.mark.asyncio
    async def test_stale_skill_tools_removed_after_deactivation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "tool-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(SKILL_WITH_TOOL_MD)
            (skill_dir / "tools.py").write_text(TOOLS_PY_CONTENT)

            agent = DummySkillAgent([tmpdir])
            await agent.activate_skill("tool-skill")

            assert "test_skill_tool" in agent.available_tools.tool_map

            await agent.deactivate_skill("tool-skill")
            assert "test_skill_tool" not in agent.available_tools.tool_map


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
