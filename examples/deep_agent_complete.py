"""
Deep Agent Complete Example - Demonstrating All 6 Core Capabilities

This example demonstrates how to use spoon-core's Deep Agent system with:
1. Agent Harness: Explicit Plan-Act-Reflect loop
2. Backends: Unified LLM provider abstraction
3. Subagents: Hierarchical agent delegation
4. HITL: Human-in-the-Loop approval for critical operations
5. Memory: Session persistence and restoration
6. Middleware: Pluggable hook system

Run the example:
    python examples/deep_agent_complete.py
"""

import asyncio
import logging
from typing import Optional, Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools.base import BaseTool
from spoon_ai.tools import ToolManager

# Import middleware system
from spoon_ai.middleware import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResult,
)

from spoon_ai.middleware.planning import PlanningMiddleware
from spoon_ai.tools.hitl import HITLMiddleware, ApprovalDecision
from spoon_ai.agents.subagents import SubAgentSpec, SubAgentMiddleware
from spoon_ai.memory.checkpointer import SQLiteCheckpointer, CheckpointMiddleware


# ============================================================================
# Simple Built-in Tools (No MCP dependency)
# ============================================================================

class CalculatorTool(BaseTool):
    """Simple calculator tool for arithmetic operations."""

    name: str = "calculator"
    description: str = "Perform arithmetic calculations. Input: expression (e.g., '2 + 3 * 4')"
    parameters: dict = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate (e.g., '2 + 3 * 4')"}
        },
        "required": ["expression"]
    }

    async def execute(self, expression: str = "", **kwargs) -> str:
        try:
            expr = expression or kwargs.get("input", "")
            # Safe eval for simple math expressions
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expr):
                return f"Error: Invalid characters in expression"
            result = eval(expr)
            return f"Result: {expr} = {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class SearchTool(BaseTool):
    """Mock search tool that simulates web search."""

    name: str = "search"
    description: str = "Search for information. Input: query string"
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string"}
        },
        "required": ["query"]
    }

    async def execute(self, query: str = "", **kwargs) -> str:
        q = query or kwargs.get("input", "")
        # Simulated search results
        results = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "ai": "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
            "agent": "An AI agent is a system that perceives its environment and takes actions to achieve goals.",
            "llm": "Large Language Models (LLMs) are neural networks trained on vast text data for language tasks.",
        }

        query_lower = q.lower()
        for key, value in results.items():
            if key in query_lower:
                return f"Search result for '{q}': {value}"

        return f"Search result for '{q}': No specific results found. This is a demonstration tool."


class NoteTool(BaseTool):
    """Tool to save and retrieve notes."""

    name: str = "note"
    description: str = "Save or retrieve notes. Input: 'save:<content>' or 'get:<key>' or 'list'"
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command: 'save:<content>' or 'get:<key>' or 'list'"}
        },
        "required": ["command"]
    }

    _notes: Dict[str, str] = {}

    async def execute(self, command: str = "", **kwargs) -> str:
        cmd = command or kwargs.get("input", "")
        if cmd.startswith("save:"):
            content = cmd[5:].strip()
            key = f"note_{len(self._notes) + 1}"
            self._notes[key] = content
            return f"Saved note as '{key}': {content}"
        elif cmd.startswith("get:"):
            key = cmd[4:].strip()
            if key in self._notes:
                return f"Note '{key}': {self._notes[key]}"
            return f"Note '{key}' not found"
        elif cmd == "list":
            if not self._notes:
                return "No notes saved"
            return "Notes: " + ", ".join(f"{k}: {v[:30]}..." for k, v in self._notes.items())
        else:
            return "Usage: 'save:<content>' or 'get:<key>' or 'list'"


class AnalysisTool(BaseTool):
    """Tool for text analysis."""

    name: str = "analyze"
    description: str = "Analyze text. Input: text to analyze"
    parameters: dict = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to analyze"}
        },
        "required": ["text"]
    }

    async def execute(self, text: str = "", **kwargs) -> str:
        t = text or kwargs.get("input", "")
        word_count = len(t.split())
        char_count = len(t)
        sentence_count = t.count('.') + t.count('!') + t.count('?')

        return f"Analysis: {word_count} words, {char_count} characters, {sentence_count} sentences"


# ============================================================================
# Custom Middleware Example
# ============================================================================

class ObservabilityMiddleware(AgentMiddleware):
    """Observability middleware - logs all calls and demonstrates AgentRuntime/AgentPhase usage"""

    system_prompt = "All your operations are logged for auditing purposes."

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.tool_count = 0
        self.phase_transitions = []

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """Intercept and log LLM calls"""
        self.call_count += 1
        print(f"\n📊 [Observability] LLM Call #{self.call_count}")

        if request.phase:
            print(f"   Phase: {request.phase.value}")
        if request.runtime:
            print(f"   Agent: {request.runtime.agent_name}, Step: {request.runtime.current_step}/{request.runtime.max_steps}")

        response = await handler(request)
        print(f"   Response: {len(response.content)} chars")
        return response

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolCallResult:
        """Intercept and log tool calls"""
        self.tool_count += 1
        print(f"\n🔧 [Observability] Tool #{self.tool_count}: {request.tool_name}")

        result = await handler(request)
        print(f"   Success: {result.success}")
        return result

    def on_plan_phase(self, runtime: AgentRuntime, phase_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        print(f"\n🎯 [PLAN Phase] Agent: {runtime.agent_name}")
        self.phase_transitions.append(("PLAN", runtime.current_step))
        return None

    def on_reflect_phase(self, runtime: AgentRuntime, phase_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        print(f"\n💭 [REFLECT Phase] Agent: {runtime.agent_name}, Step: {runtime.current_step}")
        self.phase_transitions.append(("REFLECT", runtime.current_step))
        return None

    def on_finish_phase(self, runtime: AgentRuntime, phase_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        print(f"\n🏁 [FINISH Phase] Agent: {runtime.agent_name}")
        print(f"   Phase transitions: {self.phase_transitions}")
        return None


# ============================================================================
# Full Integration Example - All 6 Core Capabilities
# ============================================================================

async def run_deep_agent():
    """Complete Deep Agent demonstrating all 6 core capabilities:

    1. Agent Harness: Plan-Act-Reflect loop (enable_plan_phase, enable_reflect_phase)
    2. Backends: Unified LLM abstraction (ChatBot with any provider)
    3. Subagents: Hierarchical delegation (SubAgentMiddleware)
    4. HITL: Human approval workflow (HITLMiddleware)
    5. Memory: Session persistence (CheckpointMiddleware + SQLiteCheckpointer)
    6. Middleware: Pluggable hooks (ObservabilityMiddleware, PlanningMiddleware)
    """
    print("\n" + "="*80)
    print("Deep Agent - 6 Core Capabilities Demo")
    print("="*80)
    print("\n1. Agent Harness: Plan-Act-Reflect loop")
    print("2. Backends: Unified LLM abstraction")
    print("3. Subagents: Hierarchical delegation")
    print("4. HITL: Human approval workflow")
    print("5. Memory: Session persistence")
    print("6. Middleware: Pluggable hooks")
    print("="*80 + "\n")

    # Auto-approval callback (for demo)
    def smart_approver(request):
        print(f"\n🤔 [HITL] Auto-approving tool: {request.tool_name}")
        return ApprovalDecision.APPROVE

    # [5. Memory] Checkpointer for session persistence
    checkpointer = SQLiteCheckpointer("deep_agent_demo.db")

    # [2. Backends] LLM provider abstraction
    llm = ChatBot()

    # Create simple tools
    calculator = CalculatorTool()
    search = SearchTool()
    note = NoteTool()
    analyze = AnalysisTool()

    # [3. Subagents] Specialized agents for delegation
    subagents = [
        SubAgentSpec(
            name="researcher",
            description="Research and information gathering specialist. Use this for searching information.",
            system_prompt="You are a research expert. Always use the search tool to find information before answering.",
            tools=[search],
            max_steps=5
        ),
        SubAgentSpec(
            name="analyst",
            description="Data analysis and calculation specialist. Use this for math and text analysis.",
            system_prompt="You are an analysis expert. Always use the calculator for math and analyze tool for text.",
            tools=[calculator, analyze],
            max_steps=5
        )
    ]

    # Tool manager with all tools
    tool_manager = ToolManager(tools=[calculator, search, note, analyze])

    # Assemble complete Deep Agent with all 6 capabilities
    agent = ToolCallAgent(
        name="deep-agent",
        llm=llm,
        thread_id="demo-session-v2",  # New session to avoid stale checkpoints
        available_tools=tool_manager,
        system_prompt="""You are a Deep Agent that demonstrates all 6 core capabilities.
You MUST use tools to answer questions - never answer from memory alone.
Available tools: search, calculator, analyze, note.
For each task, call the appropriate tool and report its output.""",

        # [6. Middleware] Complete middleware stack
        middleware=[
            ObservabilityMiddleware(),                           # Custom observability
            CheckpointMiddleware(checkpointer),                  # [5] Persistence
            PlanningMiddleware(auto_plan=True),                  # [1] Planning
            SubAgentMiddleware(subagents=subagents),             # [3] Subagents
            HITLMiddleware(                                      # [4] Human approval
                interrupt_on={"note": True},                     # Note operations need approval
                approval_callback=smart_approver
            ),
        ],

        # [1. Agent Harness] Enable Plan-Act-Reflect loop
        enable_plan_phase=True,
        enable_reflect_phase=True,
        reflect_interval=3,

        max_steps=15  # Increased for complex multi-phase task
    )

    # Execute a complex multi-step task that requires tool usage
    print("🚀 Starting multi-phase AI research task...\n")

    # Track execution for debugging
    start_time = asyncio.get_event_loop().time()

    try:
        result = await agent.run("""
        IMPORTANT: You MUST use the available tools to complete this task. Do NOT answer from memory.

        Complete these tasks step by step, using the appropriate tools for each:

        STEP 1 - Use the search tool directly:
        - Search for "AI agents"
        - Search for "LLM"

        STEP 2 - Use the calculator tool:
        - Calculate: (100 * 1.25) + (50 * 0.8) - 15
        - Calculate: 256 * 4 + 128

        STEP 3 - Use the analyze tool:
        - Analyze this text: "AI agents are autonomous systems that perceive their environment and take actions to achieve goals."

        STEP 4 - Use the note tool (requires HITL approval):
        - Save a note with key findings: "save:Research completed - AI agents use LLMs for reasoning"
        - Save another note: "save:Calculations verified - growth projections computed"

        STEP 5 - Provide final summary combining all tool outputs.

        You MUST call each tool explicitly. Show me the results from each tool call.
        """)
    except Exception as e:
        print(f"❌ Error during agent.run(): {e}")
        import traceback
        traceback.print_exc()
        result = f"Error: {e}"

    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"\n⏱️ Execution time: {elapsed:.2f}s")

    print(f"\n{'='*80}")
    print("Final Result:")
    print("="*80)
    print(result)

    # Show result statistics
    print(f"\n📈 Result Statistics:")
    print(f"   - Result length: {len(result)} chars")
    print(f"   - Lines: {result.count(chr(10)) + 1}")

    # Diagnostic information
    # Note: current_step is reset to 0 after run() completes (this is expected)
    diagnostics = agent.get_diagnostics()
    print(f"\n📊 Diagnostics (after reset):")
    print(f"   - Middleware: {diagnostics.get('middleware_count', 0)}")
    print(f"   - Max steps: {diagnostics['max_steps']}")
    print(f"   - Memory messages: {diagnostics.get('memory_messages', 0)}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run the Deep Agent example"""
    print("\n🚀 Deep Agent Example - spoon-core\n")
    print("This demo showcases all 6 core capabilities:")
    print("  • Agent Harness (Plan-Act-Reflect)")
    print("  • Backends (LLM abstraction)")
    print("  • Subagents (hierarchical delegation)")
    print("  • HITL (human approval)")
    print("  • Memory (checkpointing)")
    print("  • Middleware (pluggable hooks)")
    print()

    try:
        await run_deep_agent()
        print("\n✅ Demo Complete!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Clean up old checkpoint if it exists (for fresh demo)
    import os
    if os.path.exists("deep_agent_demo.db"):
        os.remove("deep_agent_demo.db")
        print("🗑️ Removed old checkpoint database for fresh demo")

    asyncio.run(main())
