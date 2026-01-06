"""
Spoon-Core Deep Agent Complete Example
======================================

This example demonstrates spoon-core's Deep Agent system, fully compatible
with LangChain DeepAgents architecture.

Table of Contents:
    1. Quick Start       - Minimal runnable examples
    2. HITL              - Complete interrupt/resume flow
    3. Subagents         - Hierarchical delegation with HITL inheritance
    4. Backends          - Pluggable storage backends
    5. Middleware Stack  - Complete middleware composition (incl. Summarization)
    6. Graph Workflows   - StateGraph with caching and checkpointing
    7. Full Integration  - Production-ready configuration

Run:
    python examples/deep_agent_complete.py

    # Run specific example:
    python examples/deep_agent_complete.py --example 1
    python examples/deep_agent_complete.py --example hitl

Requirements:
    - OpenAI API key (set OPENAI_API_KEY environment variable)
    - Optional: npx for MCP tools
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# Core imports
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager, BaseTool

# Middleware imports
from spoon_ai.middleware import (
    FilesystemMiddleware,
    TodoListMiddleware,
    SummarizationMiddleware,
    AnthropicPromptCachingMiddleware,
    PatchToolCallsMiddleware,
)

# HITL imports
from spoon_ai.tools.hitl import (
    HumanInTheLoopMiddleware,
    ApprovalDecision,
)

# Subagent imports
from spoon_ai.agents.subagents import (
    Command,
    SubAgentSpec,
    SubAgentMiddleware,
)

# Memory/Checkpointing
from spoon_ai.memory.checkpointer import SQLiteCheckpointer, CheckpointMiddleware

# Backends
from spoon_ai.backends import (
    create_state_backend,
    create_store_backend,
    create_composite_backend,
)

# Graph (optional)
try:
    from spoon_ai.graph import (
        StateGraph,
        END,
        InMemoryCheckpointer as GraphCheckpointer,
        InMemoryCache,
    )
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


# ============================================================================
# Helper: Create LLM instance
# ============================================================================

def create_llm(model: str = "gpt-4o-mini") -> ChatBot:
    """Create a ChatBot instance with fallback."""
    return ChatBot(provider="openai", model=model)


def section_header(title: str) -> str:
    """Generate a section header."""
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}"


# ============================================================================
# Part 1: Quick Start - Minimal Runnable Examples
# ============================================================================

async def example_1_quick_start():
    """Quick Start: Minimal examples to get started in 5 minutes."""
    print(section_header("Part 1: Quick Start - Minimal Runnable Examples"))

    llm = create_llm()

    # Example 1a: Simplest Agent
    print("\n[1a] Simplest Agent (no tools)")
    simple_agent = ToolCallAgent(name="simple", llm=llm, max_steps=3)
    result = await simple_agent.run("What is 2 + 2? Reply in one sentence.")
    print(f"  Result: {result}")

    # Example 1b: Agent with Custom Tools
    print("\n[1b] Agent with Custom Tools")

    class CalculateTool(BaseTool):
        name: str = "calculate"
        description: str = "Calculate a mathematical expression."
        parameters: dict = {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }

        async def execute(self, expression: str) -> str:
            allowed = set("0123456789+-*/(). ")
            if all(c in allowed for c in expression):
                return f"Result: {eval(expression)}"
            return "Error: Invalid expression"

    class GetCurrentTimeTool(BaseTool):
        name: str = "get_current_time"
        description: str = "Get the current time."
        parameters: dict = {"type": "object", "properties": {}}

        async def execute(self) -> str:
            from datetime import datetime
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    tool_agent = ToolCallAgent(
        name="calculator",
        llm=llm,
        available_tools=ToolManager(tools=[CalculateTool(), GetCurrentTimeTool()]),
        max_steps=5
    )
    result = await tool_agent.run("Calculate 123 * 456, then tell me the current time.")
    print(f"  Result: {result[:200]}...")
    print("\n✅ Quick Start Complete")


# ============================================================================
# Part 2: HITL - Complete Interrupt/Resume Flow
# ============================================================================

async def example_2_hitl():
    """HITL: Complete interrupt and resume workflow (documentation only)."""
    print(section_header("Part 2: HITL - Complete Interrupt/Resume Flow"))

    llm = create_llm()

    class WriteFileTool(BaseTool):
        name: str = "write_file"
        description: str = "Write content to a file."
        parameters: dict = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
        async def execute(self, path: str, content: str) -> str:
            return f"Successfully wrote to {path}"

    def format_delete_description(tool_call: Dict, state: Dict, runtime: Any) -> str:
        path = tool_call.get("args", {}).get("path", "unknown")
        return f"⚠️  DELETE FILE: {path}"

    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "write_file": True,
            "delete_file": {"allowed_decisions": ["approve", "reject"], "description": format_delete_description},
        }
    )

    agent = ToolCallAgent(
        name="hitl-demo", llm=llm,
        available_tools=ToolManager(tools=[WriteFileTool()]),
        middleware=[hitl_middleware], max_steps=5
    )

    print("""
[HITL Configuration]
  - write_file: Always interrupt
  - delete_file: With allowed_decisions and dynamic description

[Interrupt Format]
  result["__interrupt__"] = [{
      "value": {"action_requests": [...], "review_configs": [...]},
      "interrupt_id": "uuid-xxx"
  }]

[Decision Types]
  1. APPROVE: {"type": "approve"}
  2. EDIT:    {"type": "edit", "args": {"path": "/new/path"}}
  3. REJECT:  {"type": "reject", "reason": "Too risky"}

[Resume Flow]
  decisions = [{"type": "approve"}]
  result = await agent.run(Command(resume={"decisions": decisions}))
""")
    print("✅ HITL Example Complete")


# ============================================================================
# Part 3: Subagents - Hierarchical Delegation
# ============================================================================

async def example_3_subagents():
    """Subagents: Hierarchical task delegation with HITL inheritance."""
    print(section_header("Part 3: Subagents - Hierarchical Delegation"))

    llm = create_llm()

    researcher = SubAgentSpec(
        name="researcher",
        description="Expert at researching topics",
        system_prompt="You are a research specialist.",
        tools=[], max_steps=5,
        interrupt_on={"web_search": True}
    )

    analyst = SubAgentSpec(
        name="analyst",
        description="Expert at analyzing data",
        system_prompt="You are a data analyst.",
        tools=[], max_steps=5,
    )

    subagent_middleware = SubAgentMiddleware(
        subagents=[researcher, analyst],
        general_purpose_agent=True,
        default_interrupt_on={"write_file": True, "execute": {"allowed_decisions": ["approve", "reject"]}},
    )

    orchestrator = ToolCallAgent(
        name="orchestrator", llm=llm,
        middleware=[subagent_middleware], max_steps=10
    )

    print(f"""
[Subagents Configured]
  - researcher: {researcher.description} (custom interrupt_on)
  - analyst: {analyst.description} (inherits default)
  - general-purpose: Auto-created

[HITL Inheritance]
  - researcher: Uses custom interrupt_on (web_search)
  - analyst/general-purpose: Inherits default_interrupt_on

[Task Tool Usage]
  task(description="Research AI trends", subagent_type="researcher")
""")
    print("✅ Subagents Example Complete")


# ============================================================================
# Part 4: Backends - Pluggable Storage
# ============================================================================

async def example_4_backends():
    """Backends: Pluggable storage for agent file operations."""
    print(section_header("Part 4: Backends - Pluggable Storage"))

    def strip_line_number(content: str) -> str:
        """Remove line number prefix from backend read output."""
        if '\t' in content:
            return content.split('\t', 1)[1].strip()
        return content.strip()

    # StateBackend
    state_backend, _ = create_state_backend()
    state_backend.write("/notes.txt", "Hello from StateBackend!")
    content = state_backend.read("/notes.txt")
    print(f"\n[StateBackend] Write/Read: {strip_line_number(content)}")

    # StoreBackend
    db_path = "example_store.db"
    store_backend = create_store_backend(db_path=db_path)
    store_backend.write("/config.json", '{"version": "1.0"}')
    content = store_backend.read("/config.json")
    print(f"[StoreBackend] Write/Read: {strip_line_number(content)}")
    if os.path.exists(db_path):
        os.remove(db_path)

    # CompositeBackend
    state_backend, _ = create_state_backend()
    store_backend = create_store_backend(db_path=db_path)
    composite = create_composite_backend(
        default=state_backend,
        routes={"/persistent/": store_backend}
    )
    composite.write("/temp/draft.txt", "Ephemeral")
    composite.write("/persistent/settings.json", '{"theme": "dark"}')
    print("[CompositeBackend] /temp/* -> StateBackend, /persistent/* -> StoreBackend")

    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n✅ Backends Example Complete")


# ============================================================================
# Part 5: Middleware Stack - Complete Composition
# ============================================================================

async def example_5_middleware_stack():
    """Middleware Stack: Production-ready middleware composition."""
    print(section_header("Part 5: Middleware Stack - Complete Composition"))

    llm = create_llm()

    middleware_stack = [
        TodoListMiddleware(),
        FilesystemMiddleware(),
        SubAgentMiddleware(general_purpose_agent=True, default_interrupt_on={"execute": True}),
        SummarizationMiddleware(model=llm, trigger=("tokens", 100000), keep=("messages", 20)),
        AnthropicPromptCachingMiddleware(cache_system_prompt=True, cache_tools=True, unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
        HumanInTheLoopMiddleware(interrupt_on={"execute": {"allowed_decisions": ["approve", "reject"]}, "write_file": True}),
    ]

    agent = ToolCallAgent(name="production-agent", llm=llm, middleware=middleware_stack, max_steps=20)

    print("\n[Middleware Stack]")
    for i, mw in enumerate(middleware_stack, 1):
        print(f"  {i}. {mw.__class__.__name__}")

    print(f"""
[Middleware Roles]
  TodoListMiddleware:     write_todos, read_todos tools
  FilesystemMiddleware:   ls, read_file, write_file, edit_file, glob, grep, execute
  SubAgentMiddleware:     task tool for delegation
  SummarizationMiddleware: Prevent context overflow
  AnthropicPromptCachingMiddleware: Reduce costs for Anthropic
  PatchToolCallsMiddleware: Fix orphaned tool calls
  HumanInTheLoopMiddleware: Human approval for dangerous ops
""")
    print(f"✅ Agent created with {len(middleware_stack)} middleware")


# ============================================================================
# Part 6: Graph Workflows - StateGraph with Caching
# ============================================================================

async def example_6_graph():
    """Graph Workflows: StateGraph with caching and checkpointing."""
    print(section_header("Part 6: Graph Workflows - StateGraph with Caching"))

    if not GRAPH_AVAILABLE:
        print("\n⚠️  Graph module not available. Skipping.")
        return

    class WorkflowState(TypedDict):
        input: str
        analysis: str
        result: str

    def analyze_node(state: WorkflowState) -> dict:
        return {"analysis": f"Analysis of '{state['input']}'"}

    def synthesize_node(state: WorkflowState) -> dict:
        return {"result": f"Result from: {state['analysis'][:30]}..."}

    graph = StateGraph(WorkflowState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "synthesize")
    graph.add_edge("synthesize", END)

    checkpointer = GraphCheckpointer()
    cache = InMemoryCache(max_entries=100)
    compiled = graph.compile(checkpointer=checkpointer, cache=cache)

    print("\n[Graph Execution]")
    result1 = await compiled.invoke({"input": "Hello, Graph!"})
    print(f"  Call 1 (miss): {result1.get('result', 'N/A')[:50]}...")

    result2 = await compiled.invoke({"input": "Hello, Graph!"})
    print(f"  Call 2 (hit):  {result2.get('result', 'N/A')[:50]}...")

    result3 = await compiled.invoke({"input": "Different input"})
    print(f"  Call 3 (miss): {result3.get('result', 'N/A')[:50]}...")

    stats = compiled.get_cache_stats()
    print(f"\n[Cache Stats] Hits: {stats['hits']}, Misses: {stats['misses']}, Rate: {stats['hit_rate']:.0%}")
    print("\n✅ Graph Workflows Complete")


# ============================================================================
# Part 7: Full Integration - Production Configuration
# ============================================================================

async def example_7_full_integration():
    """Full Integration: Production-ready Deep Agent with all features."""
    print(section_header("Part 7: Full Integration - Production Configuration"))

    llm = create_llm()
    db_path = "integration_demo.db"

    # Clean up old checkpoint to ensure fresh demo
    if os.path.exists(db_path):
        os.remove(db_path)

    checkpointer = SQLiteCheckpointer(db_path)

    # Import MCPTool for real MCP server integration
    from spoon_ai.tools.mcp_tool import MCPTool

    # Create MCP tool using SSE transport (like deepwiki_demo.py)
    # Using DeepWiki's MCP server which is publicly available
    # IMPORTANT: Tool name must match the actual MCP server tool name
    deepwiki_tool = MCPTool(
        name="read_wiki_structure",
        description="Read repository wiki structure from DeepWiki. REQUIRES: repoName parameter in format 'owner/repo' (e.g., 'XSpoonAi/spoon-core')",
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
        max_steps=5,
        step_timeout=120.0  # 120s per step to allow subagent MCP tool calls
    )

    print(f"\n[Configuration]")
    print(f"  Agent: {agent.name}, Thread: {agent.thread_id}")
    print(f"  Subagents: researcher (with DeepWiki MCP tool)")
    print(f"  Middleware: {len(middleware)} layers")

    # Task designed to trigger subagent delegation with real MCP tool
    task = """Delegate to the researcher subagent to analyze the XSpoonAi/spoon-core repository
using the read_wiki_structure tool and provide a brief summary."""

    print(f"\n[Task] Analyze repository via researcher subagent\n")

    try:
        result = await agent.run(task, timeout=180.0)  # Increased timeout for MCP calls
        lines = result.split('\n')[:15]
        print("[Result]")
        for line in lines:
            print(f"  {line}")
        if len(result.split('\n')) > 15:
            print("  ...")
    except Exception as e:
        print(f"Error: {e}")

    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n✅ Full Integration Complete")


# ============================================================================
# Main Entry Point
# ============================================================================

EXAMPLES = {
    "1": ("Quick Start", example_1_quick_start),
    "quick": ("Quick Start", example_1_quick_start),
    "2": ("HITL", example_2_hitl),
    "hitl": ("HITL", example_2_hitl),
    "3": ("Subagents", example_3_subagents),
    "subagents": ("Subagents", example_3_subagents),
    "4": ("Backends", example_4_backends),
    "backends": ("Backends", example_4_backends),
    "5": ("Middleware Stack", example_5_middleware_stack),
    "middleware": ("Middleware Stack", example_5_middleware_stack),
    "6": ("Graph Workflows", example_6_graph),
    "graph": ("Graph Workflows", example_6_graph),
    "7": ("Full Integration", example_7_full_integration),
    "full": ("Full Integration", example_7_full_integration),
}


async def run_all_examples():
    """Run all examples in sequence."""
    print(section_header("Spoon-Core Deep Agent - Complete Examples"))
    print("   LangChain DeepAgents Compatible\n")
    print("Table of Contents:")
    print("  1. Quick Start    2. HITL           3. Subagents")
    print("  4. Backends       5. Middleware     6. Graph Workflows")
    print("  7. Full Integration")

    await example_1_quick_start()
    await example_2_hitl()
    await example_3_subagents()
    await example_4_backends()
    await example_5_middleware_stack()
    await example_6_graph()
    await example_7_full_integration()

    print(section_header("✅ All Examples Complete!"))


async def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Spoon-Core Deep Agent Examples")
    parser.add_argument("-e", "--example", type=str, help="Run specific example (1-7)")
    parser.add_argument("-l", "--list", action="store_true", help="List available examples")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable examples:")
        seen = set()
        for key, (name, _) in EXAMPLES.items():
            if name not in seen:
                print(f"  {key}: {name}")
                seen.add(name)
        return

    if args.example:
        key = args.example.lower()
        if key in EXAMPLES:
            name, func = EXAMPLES[key]
            print(f"\n🚀 Running: {name}")
            await func()
        else:
            print(f"Unknown example: {args.example}. Use --list to see options.")
            sys.exit(1)
    else:
        await run_all_examples()


if __name__ == "__main__":
    asyncio.run(main())
