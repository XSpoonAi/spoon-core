"""Real ReAct + Skill demo using ops_tools.

Run:

    cd ~/project/xspoonai/spoon-core
    source .venv/bin/activate
    python examples/ops_react_demo.py

What it does:
- Uses SpoonReactSkill (ReAct agent + Skill system)
- Loads example skills from ./examples/skills
- Injects ops tools (local_exec + fs_* confined to workspace root)
- Forces the agent to:
  1) write input JSON to a file via fs_write
  2) call the skill script tool run_script_data-processor_analyze (real Skill tool)
  3) write a markdown report via fs_write

Notes:
- Requires real LLM credentials; loads .env automatically.
"""

import asyncio
import os
from pathlib import Path

from spoon_ai.agents import SpoonReactSkill
from spoon_ai.tools.ops_tools import build_ops_tools


def _load_dotenv_if_present(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
        return

    load_dotenv(env_path)


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv_if_present(root)

    agent = SpoonReactSkill(
        name="ops_demo",
        skill_paths=[str(root / "examples" / "skills")],
        max_steps=8,
        tool_choice="required",
    )

    # Inject ops tools
    for t in build_ops_tools(root_dir=str(root), allow_exec=["python", "echo", "ls", "cat", "pytest"]):
        agent.available_tools.add_tool(t)

    # Ensure the skill tool is definitely present
    await agent.activate_skill("data-processor")

    # Refresh prompts so ReAct sees tools
    agent._refresh_prompts()

    data = [
        {"user": "alice", "score": 10},
        {"user": "bob", "score": 15},
        {"user": "alice", "score": 7},
    ]

    # IMPORTANT: the skill script analyze.py expects valid JSON via stdin.
    # So we provide a proper JSON string (not Python repr).
    import json

    json_text = json.dumps(data)

    request = f"""
You MUST follow these steps exactly:

1) Use fs_write to write the following JSON into examples/out/input.json
2) Use the tool run_script_data-processor_analyze to analyze the JSON.
   Pass EXACTLY the JSON text (not Python repr) as the tool input.
3) Use fs_write to write a markdown report to examples/out/ops_demo_report.md.
   Include: row count, unique users, average score.

JSON:
{json_text}
""".strip()

    print(await agent.run(request))


if __name__ == "__main__":
    asyncio.run(main())
