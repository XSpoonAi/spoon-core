"""Smoke test: real SpoonReactSkill + SkillTool injection + ops tools.

No network/LLM call. Only validates wiring.
"""

import asyncio
from pathlib import Path

from spoon_ai.agents import SpoonReactSkill
from spoon_ai.tools.ops_tools import build_ops_tools


def test_react_skill_injection_smoke():
    root = Path(__file__).resolve().parents[1]
    skill_path = root / "examples" / "skills"
    assert skill_path.exists(), f"expected {skill_path} to exist"

    agent = SpoonReactSkill(
        name="smoke",
        skill_paths=[str(skill_path)],
        max_steps=1,
    )

    for t in build_ops_tools(root_dir=str(root), allow_exec=["echo"]):
        agent.available_tools.add_tool(t)

    asyncio.run(agent.activate_skill("data-processor"))

    tool_names = set(agent.available_tools.tool_map.keys())

    assert "fs_write" in tool_names
    assert "fs_read" in tool_names
    assert "fs_list" in tool_names

    assert any(name.startswith("run_script_data-processor_") for name in tool_names)
