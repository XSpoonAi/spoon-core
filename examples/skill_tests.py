#!/usr/bin/env python3
"""
Skill System Integration Tests

Comprehensive test suite for the SpoonAI skill system including:
- Skill models and data structures
- Script execution system
- Skill loader and discovery
- Skill manager lifecycle
- Script tools integration
- Trigger matching
- Multi-skill workflows

Usage:
    python examples/skill_tests.py
    python examples/skill_tests.py --verbose
    python examples/skill_tests.py --test models
    python examples/skill_tests.py --test executor
    python examples/skill_tests.py --test manager
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Path to example skills
EXAMPLES_SKILLS_PATH = Path(__file__).parent / "skills"

from spoon_ai.skills import (
    SkillManager,
    SkillLoader,
    ScriptExecutor,
    ScriptType,
    SkillScript,
    ScriptConfig,
    ScriptResult,
    get_executor,
    set_scripts_enabled,
    ScriptTool,
    create_script_tools,
)


class TestResult:
    """Test result container."""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


class SkillTester:
    """Comprehensive test runner for skill system."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"  [DEBUG] {message}")

    def report(self, name: str, passed: bool, message: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        emoji = "✓" if passed else "✗"
        print(f"  {emoji} {name}: {status}")
        if message and (not passed or self.verbose):
            print(f"    {message}")
        self.results.append(TestResult(name, passed, message))

    # =========================================================================
    # MODEL TESTS
    # =========================================================================

    async def test_models(self) -> None:
        """Test skill and script models."""
        print("\n=== Testing Models ===")

        # Test ScriptType enum
        try:
            assert ScriptType.PYTHON.value == "python"
            assert ScriptType.SHELL.value == "shell"
            assert ScriptType.BASH.value == "bash"
            self.report("ScriptType enum", True)
        except Exception as e:
            self.report("ScriptType enum", False, str(e))

        # Test SkillScript with file
        try:
            script = SkillScript(name="test", type=ScriptType.PYTHON, file="test.py")
            assert script.name == "test"
            assert script.file == "test.py"
            self.report("SkillScript with file", True)
        except Exception as e:
            self.report("SkillScript with file", False, str(e))

        # Test SkillScript with inline
        try:
            script = SkillScript(name="inline", type=ScriptType.BASH, inline="echo hello")
            assert script.inline == "echo hello"
            self.report("SkillScript with inline", True)
        except Exception as e:
            self.report("SkillScript with inline", False, str(e))

        # Test validation - requires source
        try:
            SkillScript(name="no-source", type=ScriptType.PYTHON)
            self.report("SkillScript validation (no source)", False, "Should have raised error")
        except ValueError:
            self.report("SkillScript validation (no source)", True)
        except Exception as e:
            self.report("SkillScript validation (no source)", False, str(e))

        # Test validation - exclusive source
        try:
            SkillScript(name="both", type=ScriptType.PYTHON, file="a.py", inline="print()")
            self.report("SkillScript validation (exclusive)", False, "Should have raised error")
        except ValueError:
            self.report("SkillScript validation (exclusive)", True)
        except Exception as e:
            self.report("SkillScript validation (exclusive)", False, str(e))

        # Test ScriptConfig
        try:
            config = ScriptConfig(
                enabled=True,
                definitions=[
                    SkillScript(name="s1", type=ScriptType.PYTHON, file="s1.py"),
                    SkillScript(name="s2", type=ScriptType.BASH, inline="echo test"),
                ]
            )
            assert config.enabled is True
            assert len(config.definitions) == 2
            assert config.get_script("s1") is not None
            assert config.get_script("nonexistent") is None
            self.report("ScriptConfig", True)
        except Exception as e:
            self.report("ScriptConfig", False, str(e))

        # Test ScriptResult
        try:
            result = ScriptResult(script_name="test", success=True, exit_code=0, stdout="output")
            assert result.success is True
            assert "output" in result.to_string()

            failed = ScriptResult(script_name="test", success=False, error="failed")
            assert "Error" in failed.to_string()
            self.report("ScriptResult", True)
        except Exception as e:
            self.report("ScriptResult", False, str(e))

    # =========================================================================
    # EXECUTOR TESTS
    # =========================================================================

    async def test_executor(self) -> None:
        """Test script executor."""
        print("\n=== Testing Executor ===")

        # Test executor creation
        try:
            executor = ScriptExecutor(enabled=True, default_timeout=30)
            assert executor.enabled is True
            self.report("Executor creation", True)
        except Exception as e:
            self.report("Executor creation", False, str(e))

        # Test interpreter detection
        try:
            executor = ScriptExecutor()
            assert executor.is_available(ScriptType.PYTHON)
            self.log(f"Python interpreter: {executor.get_interpreter(ScriptType.PYTHON)}")
            self.report("Interpreter detection", True)
        except Exception as e:
            self.report("Interpreter detection", False, str(e))

        # Test inline Python execution
        try:
            executor = ScriptExecutor(enabled=True)
            script = SkillScript(name="hello", type=ScriptType.PYTHON, inline="print('Hello!')")
            result = await executor.execute(script)
            assert result.success is True
            assert "Hello!" in result.stdout
            self.report("Inline Python execution", True)
        except Exception as e:
            self.report("Inline Python execution", False, str(e))

        # Test stdin input
        try:
            executor = ScriptExecutor(enabled=True)
            script = SkillScript(
                name="echo",
                type=ScriptType.PYTHON,
                inline="import sys; print(f'Got: {sys.stdin.read().strip()}')"
            )
            result = await executor.execute(script, input_text="test data")
            assert result.success is True
            assert "Got: test data" in result.stdout
            self.report("Stdin input", True)
        except Exception as e:
            self.report("Stdin input", False, str(e))

        # Test disabled executor
        try:
            executor = ScriptExecutor(enabled=False)
            script = SkillScript(name="disabled", type=ScriptType.PYTHON, inline="print('no')")
            result = await executor.execute(script)
            assert result.success is False
            assert "disabled" in result.error.lower()
            self.report("Disabled executor", True)
        except Exception as e:
            self.report("Disabled executor", False, str(e))

        # Test timeout
        try:
            executor = ScriptExecutor(enabled=True)
            script = SkillScript(
                name="slow",
                type=ScriptType.PYTHON,
                inline="import time; time.sleep(10)",
                timeout=1
            )
            result = await executor.execute(script)
            assert result.success is False
            assert "timed out" in result.error.lower()
            self.report("Timeout handling", True)
        except Exception as e:
            self.report("Timeout handling", False, str(e))

        # Test global executor functions
        try:
            executor = get_executor()
            assert executor is not None
            set_scripts_enabled(False)
            assert executor.enabled is False
            set_scripts_enabled(True)
            assert executor.enabled is True
            self.report("Global executor functions", True)
        except Exception as e:
            self.report("Global executor functions", False, str(e))

    # =========================================================================
    # SCRIPT TOOL TESTS
    # =========================================================================

    async def test_script_tool(self) -> None:
        """Test script tool wrapper."""
        print("\n=== Testing ScriptTool ===")

        # Test tool creation
        try:
            script = SkillScript(
                name="analyze",
                description="Analyze data",
                type=ScriptType.PYTHON,
                inline="print('analyzing')"
            )
            tool = ScriptTool(script=script, skill_name="test-skill")
            assert "run_script_test-skill_analyze" in tool.name
            assert "Analyze data" in tool.description
            self.report("ScriptTool creation", True)
        except Exception as e:
            self.report("ScriptTool creation", False, str(e))

        # Test tool execution
        try:
            script = SkillScript(name="greet", type=ScriptType.PYTHON, inline="print('Hello!')")
            tool = ScriptTool(script=script, skill_name="test")
            result = await tool.execute()
            assert "Hello!" in result
            self.report("ScriptTool execution", True)
        except Exception as e:
            self.report("ScriptTool execution", False, str(e))

        # Test create_script_tools
        try:
            scripts = [
                SkillScript(name="s1", type=ScriptType.PYTHON, inline="print(1)"),
                SkillScript(name="s2", type=ScriptType.PYTHON, inline="print(2)"),
            ]
            tools = create_script_tools("test-skill", scripts)
            assert len(tools) == 2
            assert all(isinstance(t, ScriptTool) for t in tools)
            self.report("create_script_tools", True)
        except Exception as e:
            self.report("create_script_tools", False, str(e))

    # =========================================================================
    # LOADER TESTS
    # =========================================================================

    async def test_loader(self) -> None:
        """Test skill loader and discovery."""
        print("\n=== Testing Loader ===")

        # Test discovery
        try:
            loader = SkillLoader(additional_paths=[EXAMPLES_SKILLS_PATH])
            skills = loader.load_all()
            self.log(f"Loaded {len(skills)} skills: {list(skills.keys())}")
            assert len(skills) > 0
            self.report("Skill discovery", True, f"Found {len(skills)} skills")
        except Exception as e:
            self.report("Skill discovery", False, str(e))

        # Test expected skills exist
        try:
            loader = SkillLoader(additional_paths=[EXAMPLES_SKILLS_PATH])
            loader.load_all()
            expected = ["research", "data-processor", "web3-research"]
            found = []
            for name in expected:
                skill = loader.get_skill(name)
                if skill:
                    found.append(name)
                    self.report(f"Skill '{name}' discovered", True)
                else:
                    self.report(f"Skill '{name}' discovered", False, "Not found")
        except Exception as e:
            self.report("Expected skills", False, str(e))

        # Test skill metadata
        try:
            loader = SkillLoader(additional_paths=[EXAMPLES_SKILLS_PATH])
            loader.load_all()

            research = loader.get_skill("research")
            if research:
                assert research.metadata.name == "research"
                assert len(research.metadata.triggers) > 0
                self.report("Research skill metadata", True)

            dp = loader.get_skill("data-processor")
            if dp:
                assert dp.metadata.has_scripts()
                assert dp.metadata.scripts_enabled()
                self.report("Data-processor skill metadata", True, f"Scripts: {dp.script_names}")

            web3 = loader.get_skill("web3-research")
            if web3:
                self.report("Web3-research skill metadata", True)
        except Exception as e:
            self.report("Skill metadata", False, str(e))

    # =========================================================================
    # MANAGER TESTS
    # =========================================================================

    async def test_manager(self) -> None:
        """Test skill manager lifecycle."""
        print("\n=== Testing Manager Lifecycle ===")

        # Test activation
        try:
            manager = SkillManager(
                skill_paths=[str(EXAMPLES_SKILLS_PATH)],
                auto_discover=True,
                scripts_enabled=True
            )
            skill = await manager.activate("data-processor", {"format": "json"})
            assert skill is not None
            assert manager.is_active("data-processor")
            self.report("Skill activation", True)
        except Exception as e:
            self.report("Skill activation", False, str(e))

        # Test context injection
        try:
            manager = SkillManager(
                skill_paths=[str(EXAMPLES_SKILLS_PATH)],
                auto_discover=True,
                scripts_enabled=True
            )
            await manager.activate("data-processor")
            context = manager.get_active_context()
            assert "data-processor" in context.lower() or "Data Processor" in context
            self.report("Context injection", True)
        except Exception as e:
            self.report("Context injection", False, str(e))

        # Test script tools created
        try:
            manager = SkillManager(
                skill_paths=[str(EXAMPLES_SKILLS_PATH)],
                auto_discover=True,
                scripts_enabled=True
            )
            await manager.activate("data-processor")
            tools = manager.get_active_tools()
            script_tools = [t for t in tools if "run_script" in t.name]
            self.log(f"Script tools: {[t.name for t in script_tools]}")
            self.report("Script tools created", len(script_tools) > 0, f"Found {len(script_tools)}")
        except Exception as e:
            self.report("Script tools created", False, str(e))

        # Test deactivation
        try:
            manager = SkillManager(
                skill_paths=[str(EXAMPLES_SKILLS_PATH)],
                auto_discover=True,
                scripts_enabled=True
            )
            await manager.activate("data-processor")
            await manager.deactivate("data-processor")
            assert not manager.is_active("data-processor")
            self.report("Skill deactivation", True)
        except Exception as e:
            self.report("Skill deactivation", False, str(e))

        # Test script tools cleaned up
        try:
            manager = SkillManager(
                skill_paths=[str(EXAMPLES_SKILLS_PATH)],
                auto_discover=True,
                scripts_enabled=True
            )
            await manager.activate("data-processor")
            await manager.deactivate("data-processor")
            tools_after = manager.get_script_tools("data-processor")
            self.report("Script tools cleaned up", len(tools_after) == 0)
        except Exception as e:
            self.report("Script tools cleaned up", False, str(e))

    # =========================================================================
    # MULTIPLE SKILLS TESTS
    # =========================================================================

    async def test_multiple_skills(self) -> None:
        """Test multiple skills activation."""
        print("\n=== Testing Multiple Skills ===")

        try:
            manager = SkillManager(
                skill_paths=[str(EXAMPLES_SKILLS_PATH)],
                auto_discover=True,
                scripts_enabled=True
            )
            await manager.activate("research")
            await manager.activate("data-processor")

            active = manager.get_active_skill_names()
            self.log(f"Active skills: {active}")
            assert "research" in active
            assert "data-processor" in active
            self.report("Multiple skills active", True, f"Active: {active}")

            context = manager.get_active_context()
            self.report("Combined context generated", len(context) > 100)

            count = await manager.deactivate_all()
            self.report("Deactivate all", count == 2, f"Deactivated {count}")
        except Exception as e:
            self.report("Multiple skills", False, str(e))

    # =========================================================================
    # SCRIPT EXECUTION TESTS
    # =========================================================================

    async def test_script_execution(self) -> None:
        """Test script execution through manager."""
        print("\n=== Testing Script Execution Integration ===")

        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        await manager.activate("data-processor")

        # Test analyze script
        try:
            result = await manager.execute_script(
                "data-processor", "analyze",
                input_text='{"users": [{"name": "Alice"}, {"name": "Bob"}]}'
            )
            self.log(f"Analyze result: {result.stdout[:200]}...")
            self.report("Execute analyze script", result.success)
        except Exception as e:
            self.report("Execute analyze script", False, str(e))

        # Test transform script
        try:
            result = await manager.execute_script(
                "data-processor", "transform",
                input_text='name,age\nAlice,30\nBob,25'
            )
            self.log(f"Transform result: {result.stdout[:200]}...")
            self.report("Execute transform script", result.success)
        except Exception as e:
            self.report("Execute transform script", False, str(e))

        await manager.deactivate_all()

    # =========================================================================
    # TRIGGER MATCHING TESTS
    # =========================================================================

    async def test_trigger_matching(self) -> None:
        """Test skill trigger matching."""
        print("\n=== Testing Trigger Matching ===")

        manager = SkillManager(skill_paths=[str(EXAMPLES_SKILLS_PATH)], auto_discover=True)

        try:
            matches = manager.match_triggers("I want to research blockchain technology")
            self.log(f"Matches for 'research blockchain': {[m.name for m in matches]}")
            research_found = any(m.name == "research" for m in matches)
            self.report("Keyword trigger (research)", research_found)
        except Exception as e:
            self.report("Keyword trigger (research)", False, str(e))

        try:
            matches = manager.match_triggers("Please analyze this data and convert it to JSON")
            self.log(f"Matches for 'analyze data convert': {[m.name for m in matches]}")
            dp_found = any(m.name == "data-processor" for m in matches)
            self.report("Pattern trigger (data-processor)", dp_found)
        except Exception as e:
            self.report("Pattern trigger (data-processor)", False, str(e))

        try:
            matches = manager.match_triggers("What is the current price of Ethereum and DeFi yields?")
            self.log(f"Matches for 'Ethereum DeFi': {[m.name for m in matches]}")
            web3_found = any(m.name == "web3-research" for m in matches)
            self.report("Keyword trigger (web3-research)", web3_found)
        except Exception as e:
            self.report("Keyword trigger (web3-research)", False, str(e))

    # =========================================================================
    # STATS AND INFO TESTS
    # =========================================================================

    async def test_stats_and_info(self) -> None:
        """Test skill statistics and info."""
        print("\n=== Testing Stats and Info ===")

        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )

        # Initial stats
        try:
            stats = manager.get_stats()
            self.log(f"Initial stats: {stats}")
            assert stats["total_skills"] >= 3
            assert stats["scripts_enabled"] is True
            self.report("Initial stats", True, f"Total: {stats['total_skills']}")
        except Exception as e:
            self.report("Initial stats", False, str(e))

        # Stats after activation
        try:
            await manager.activate("data-processor")
            stats = manager.get_stats()
            assert stats["active_skills"] == 1
            assert "data-processor" in stats["active_skill_names"]
            self.report("Stats after activation", True)
        except Exception as e:
            self.report("Stats after activation", False, str(e))

        # Script execution stats
        try:
            await manager.execute_script("data-processor", "analyze", input_text="{}")
            stats = manager.get_stats()
            self.log(f"Script stats: {stats['script_execution_stats']}")
            self.report("Script execution stats tracked", stats["script_execution_stats"]["total"] > 0)
        except Exception as e:
            self.report("Script execution stats tracked", False, str(e))

        # Skill info
        try:
            info = manager.get_skill_info("data-processor")
            assert info["name"] == "data-processor"
            assert info["has_scripts"] is True
            self.report("Skill info", True, f"Scripts: {info['script_names']}")
        except Exception as e:
            self.report("Skill info", False, str(e))

        await manager.deactivate_all()

    # =========================================================================
    # SCRIPTS ENABLE/DISABLE TESTS
    # =========================================================================

    async def test_scripts_toggle(self) -> None:
        """Test enabling/disabling script execution."""
        print("\n=== Testing Scripts Enable/Disable ===")

        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )

        try:
            assert manager.get_stats()["scripts_enabled"] is True
            self.report("Scripts initially enabled", True)
        except Exception as e:
            self.report("Scripts initially enabled", False, str(e))

        try:
            manager.set_scripts_enabled(False)
            assert manager.get_stats()["scripts_enabled"] is False
            self.report("Scripts disabled", True)
        except Exception as e:
            self.report("Scripts disabled", False, str(e))

        try:
            await manager.activate("data-processor")
            result = await manager.execute_script("data-processor", "analyze", input_text="{}")
            assert result.success is False
            self.report("Script blocked when disabled", True)
        except Exception as e:
            self.report("Script blocked when disabled", False, str(e))

        try:
            manager.set_scripts_enabled(True)
            result = await manager.execute_script("data-processor", "analyze", input_text="{}")
            assert result.success is True
            self.report("Script works after re-enable", True)
        except Exception as e:
            self.report("Script works after re-enable", False, str(e))

        await manager.deactivate_all()

    # =========================================================================
    # DATA-PROCESSOR DETAILED TESTS
    # =========================================================================

    async def test_data_processor(self) -> None:
        """Test data-processor scripts with various inputs."""
        print("\n=== Testing Data-Processor Scripts ===")

        manager = SkillManager(
            skill_paths=[str(EXAMPLES_SKILLS_PATH)],
            auto_discover=True,
            scripts_enabled=True
        )
        await manager.activate("data-processor")

        test_cases = [
            ("Analyze JSON object", "analyze", '{"name": "test", "values": [1, 2, 3]}', "json"),
            ("Analyze JSON array", "analyze", '[{"id": 1}, {"id": 2}]', None),
            ("Analyze plain text", "analyze", 'Line 1\nLine 2\nLine 3', "text"),
            ("Transform CSV to JSON", "transform", 'name,age\nAlice,30\nBob,25', "csv"),
            ("Transform JSON to CSV", "transform", '[{"name": "Alice"}, {"name": "Bob"}]', "json"),
        ]

        for test_name, script, input_text, check_word in test_cases:
            try:
                result = await manager.execute_script("data-processor", script, input_text=input_text)
                passed = result.success
                if check_word:
                    passed = passed and check_word in result.stdout.lower()
                self.report(test_name, passed)
            except Exception as e:
                self.report(test_name, False, str(e))

        await manager.deactivate_all()

    # =========================================================================
    # RUN ALL
    # =========================================================================

    async def run_all(self) -> Tuple[int, int]:
        """Run all tests."""
        print("\n" + "=" * 60)
        print("Skill System Integration Tests")
        print("=" * 60)

        await self.test_models()
        await self.test_executor()
        await self.test_script_tool()
        await self.test_loader()
        await self.test_manager()
        await self.test_multiple_skills()
        await self.test_script_execution()
        await self.test_trigger_matching()
        await self.test_stats_and_info()
        await self.test_scripts_toggle()
        await self.test_data_processor()

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print("\n" + "=" * 60)
        print(f"Results: {passed} passed, {failed} failed, {len(self.results)} total")
        print("=" * 60)

        return passed, failed


async def main():
    parser = argparse.ArgumentParser(description="Skill System Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", "-t", type=str, help="Run specific test")
    args = parser.parse_args()

    tester = SkillTester(verbose=args.verbose)

    if args.test:
        test_method = getattr(tester, f"test_{args.test}", None)
        if test_method:
            print(f"\nRunning specific test: {args.test}")
            await test_method()
            passed = sum(1 for r in tester.results if r.passed)
            failed = sum(1 for r in tester.results if not r.passed)
            print(f"\nResults: {passed} passed, {failed} failed")
            sys.exit(0 if failed == 0 else 1)
        else:
            print(f"Unknown test: {args.test}")
            print("Available: models, executor, script_tool, loader, manager, multiple_skills,")
            print("           script_execution, trigger_matching, stats_and_info, scripts_toggle, data_processor")
            sys.exit(1)
    else:
        passed, failed = await tester.run_all()
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
