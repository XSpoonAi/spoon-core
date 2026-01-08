import asyncio
import logging
import os

from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.tools.base import BaseTool
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.middleware import (
    FilesystemMiddleware,
    TodoListMiddleware,
    SummarizationMiddleware,
    PatchToolCallsMiddleware,
)
from spoon_ai.agents.subagents import SubAgentSpec, SubAgentMiddleware
from spoon_ai.memory.checkpointer import SQLiteCheckpointer, CheckpointMiddleware
from spoon_ai.tools.hitl import HumanInTheLoopMiddleware
from spoon_ai.graph import InMemoryCheckpointer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)


def create_llm(model: str = "gpt-4o-mini") -> ChatBot:
    """Create a ChatBot instance with fallback."""
    return ChatBot(provider="openai", model=model)

# ============================================================================
# Custom Tool for HITL Demo
# ============================================================================

class ExecuteCommandTool(BaseTool):
    """A dangerous tool that executes shell commands - requires HITL approval."""

    def __init__(self):
        super().__init__(
            name="execute_command",
            description="Execute a shell command on the system. This is a dangerous operation that requires human approval.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        )

    async def execute(self, command: str) -> str:
        """Execute the command (simulated for safety)."""
        # For demo purposes, we simulate command execution
        return f"[SIMULATED] Command executed: {command}\nOutput: Command completed successfully."


# ============================================================================
# HITL Demo (LangChain Compatible)
# ============================================================================

async def example_hitl_demo():
    """Demonstrate Human-in-the-Loop approval workflow.

    This demo uses return value mode (__interrupt__) and Command(resume=...)
    pattern, compatible with LangChain DeepAgents.
    """
    print("\n" + "=" * 70)
    print("Human-in-the-Loop (HITL) Demo - LangChain Compatible")
    print("=" * 70)

    llm = create_llm()

    # Create the dangerous tool
    execute_tool = ExecuteCommandTool()

    # Create checkpointer (REQUIRED for HITL)
    checkpointer = InMemoryCheckpointer()

    # Configure HITL middleware
    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "execute_command": {
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        },
        checkpointer=checkpointer
    )

    # Create agent with HITL middleware
    thread_id = "hitl-demo-thread"
    agent = ToolCallAgent(
        name="hitl-demo-agent",
        llm=llm,
        tools=[execute_tool],
        middleware=[hitl_middleware, PatchToolCallsMiddleware()],
        thread_id=thread_id,
        max_steps=5,
    )

    print(f"\n  Agent: {agent.name}")
    print(f"  Thread ID: {thread_id}")
    print(f"  Tools: {[t.name for t in agent.tools]}")
    print("  HITL Protected Tools: execute_command")
    print("  Checkpointer: InMemoryCheckpointer")

    # Task that will trigger HITL
    task = "Please execute the command 'echo Hello World' to test the system."
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n[Task] {task}\n")

    # First attempt - will return __interrupt__
    print("[Step 1] Agent attempts to execute command...")
    result = await agent.run(task, timeout=60.0)

    # Check if execution was interrupted
    if isinstance(result, dict) and result.get("__interrupt__"):
        print("\n[HITL INTERRUPT] Tool execution requires approval!")
        print("-" * 50)

        # Extract interrupt information (LangChain format)
        interrupts = result["__interrupt__"][0]["value"]
        action_requests = interrupts["action_requests"]
        review_configs = interrupts["review_configs"]

        # Create a lookup map from tool name to review config
        config_map = {cfg["action_name"]: cfg for cfg in review_configs}

        # Display the pending actions to the user
        for i, action in enumerate(action_requests):
            review_config = config_map.get(action["name"], {})
            print(f"\n  Action #{i + 1}:")
            print(f"    Tool: {action['name']}")
            print(f"    Args: {action['args']}")
            print(f"    Allowed Decisions: {review_config.get('allowed_decisions', [])}")

        print("\n" + "-" * 50)

        # Simulate user approval decision
        print("\n[Simulating User Decision] Approving the command...")

        # Get user decisions (one per action_request, in order)
        decisions = [
            {"type": "approve"}  # User approved the command
        ]

        # Resume execution with decisions using Command pattern
        print("\n[Step 2] Resuming with Command(resume=...)...")
        hitl_middleware.set_resume_data({"decisions": decisions}, thread_id=thread_id)
        result = await agent.run(task, timeout=60.0)

        print("\n[Result after approval]")
        if isinstance(result, str):
            for line in result.split('\n')[:10]:
                print(f"  {line}")
        else:
            print(f"  {result}")
    else:
        # No interrupt, command executed directly
        print("\n[Result]")
        if isinstance(result, str):
            for line in result.split('\n')[:10]:
                print(f"  {line}")
        else:
            print(f"  {result}")

    print("\n" + "=" * 70)
    print("HITL Demo Complete")
    print("=" * 70)

async def example_full_integration():

    llm = create_llm()
    db_path = "integration_demo.db"

    # Clean up old checkpoint to ensure fresh demo
    if os.path.exists(db_path):
        os.remove(db_path)

    checkpointer = SQLiteCheckpointer(db_path)
    deepwiki_tool = MCPTool(
        name="read_wiki_structure",
        description="Read repository wiki structure from DeepWiki. Returns the table of contents for a GitHub repository's wiki.",
        parameters={
            "type": "object",
            "properties": {
                "repoName": {
                    "type": "string",
                    "description": "The repository name in format 'owner/repo' (e.g., 'XSpoonAi/spoon-core')"
                }
            },
            "required": ["repoName"]
        },
        mcp_config={
            "url": "https://mcp.deepwiki.com/sse",
            "transport": "sse",
            "timeout": 30,
            "headers": {
                "User-Agent": "SpoonOS-SSE-MCP/1.0",
                "Accept": "text/event-stream"
            }
        }
    )

    # Subagents with real MCP tools
    subagents = [
        SubAgentSpec(
            name="researcher",
            description="Repository researcher. Use this agent to analyze GitHub repositories using DeepWiki.",
            system_prompt="""You are a research specialist with access to the DeepWiki MCP tool.

To analyze a repository, call the 'read_wiki_structure' tool with the repoName parameter:
- repoName: The repository in format 'owner/repo' (e.g., 'XSpoonAi/spoon-core')

IMPORTANT: Always include repoName when calling read_wiki_structure.""",
            tools=[deepwiki_tool],
            max_steps=3,
            timeout=90.0  # 90s total for subagent, 30s per step
        ),
    ]

    middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(),
        SubAgentMiddleware(subagents=subagents, general_purpose_agent=True,
                          default_interrupt_on={"execute": True, "write_file": True}),
        SummarizationMiddleware(model=llm, trigger=("tokens", 100000), keep=("messages", 20)),
        CheckpointMiddleware(checkpointer),
        PatchToolCallsMiddleware(),
    ]

    agent = ToolCallAgent(
        name="deep-agent", llm=llm,
        thread_id="integration-demo-001",
        middleware=middleware,
        max_steps=10,  # Increased for complex multi-step tasks
        step_timeout=120.0  # 120s per step to allow subagent MCP tool calls
    )

    print(f"  Agent: {agent.name}, Thread: {agent.thread_id}")
    print("  Subagents: researcher (with DeepWiki MCP tool)")
    print(f"  Middleware: {len(middleware)} layers")

    # Task designed to trigger subagent delegation with real MCP tool
    # Complex multi-step task to trigger TodoList functionality
    task = """Please complete the following multi-step analysis task. Use the todo list to track your progress:

1. First, use the researcher subagent to analyze the XSpoonAi/spoon-core repository using read_wiki_structure tool
2. Based on the wiki structure, identify and list the top 3 most important core components
3. For each core component, explain its purpose and how it integrates with other parts
4. Provide a final recommendation on which component a new developer should learn first
5. Summarize the overall architecture in a brief technical overview

Make sure to update the todo list as you complete each step."""

    print("\n[Task] Multi-step repository analysis with TodoList tracking\n")

    result = await agent.run(task, timeout=300.0)  # Increased timeout for complex task
    lines = result.split('\n')[:50]  # Show more output for complex task
    print("[Result]")
    for line in lines:
        print(f"  {line}")
    if len(result.split('\n')) > 50:
        print("  ...")

    if os.path.exists(db_path):
        os.remove(db_path)

async def main():
    # Run HITL demo first (shorter)
    await example_hitl_demo()

    # Then run full integration demo
    await example_full_integration()


if __name__ == "__main__":
    asyncio.run(main())
