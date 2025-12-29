"""
Complete Deep Agent Example - Demonstrating All 6 Core Capabilities

This example demonstrates how to use spoon-core's Deep Agent system to implement:
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
import os
from typing import Optional, Any
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools.base import BaseTool
from spoon_ai.tools import ToolManager
from spoon_ai.tools.mcp_tool import MCPTool

# Import middleware system
from spoon_ai.middleware import (
    AgentMiddleware,
    AgentRuntime,
    AgentPhase,
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
# MCP Tool Creation - Lightweight External Tool Integration
# ============================================================================

async def create_deepwiki_tool() -> MCPTool:
    """Create DeepWiki repository analysis tool using MCP SSE server.

    No API key required - uses public DeepWiki MCP service.

    Available tools:
    - ask_question: Ask any question about a GitHub repository
    - read_wiki_structure: Get documentation topics
    - read_wiki_contents: View documentation
    """
    tool = MCPTool(
        name="ask_question",  # Auto-updated from MCP server
        description="Ask questions about GitHub repositories via DeepWiki",
        mcp_config={
            "url": "https://mcp.deepwiki.com/sse",
            "transport": "sse",
            "timeout": 60,
            "headers": {
                "User-Agent": "Spoon-Deep-Agent/1.0",
                "Accept": "text/event-stream"
            }
        }
    )
    # CRITICAL FIX: Load parameters before returning so LLM gets correct schema
    await tool.ensure_parameters_loaded()
    return tool

async def create_filesystem_tool() -> MCPTool:
    """Create filesystem tool using MCP filesystem server.

    No API key required - runs locally via npx.
    """
    tool = MCPTool(
        name="read_file",
        description="Read and analyze files via MCP filesystem server",
        mcp_config={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        }
    )
    # CRITICAL FIX: Load parameters before returning so LLM gets correct schema
    await tool.ensure_parameters_loaded()
    return tool

async def create_memory_tool() -> MCPTool:
    """Create memory tool using MCP memory server.

    No API key required - runs locally via npx.
    """
    tool = MCPTool(
        name="create_entities",
        description="Store and manage data via MCP memory server",
        mcp_config={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
        }
    )
    # CRITICAL FIX: Load parameters before returning so LLM gets correct schema
    await tool.ensure_parameters_loaded()
    return tool


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
        """Intercept and log LLM calls - demonstrates AgentPhase usage"""
        self.call_count += 1
        print(f"\n📊 [Observability] LLM Call #{self.call_count}")

        # ✅ USE AgentPhase: Show which phase the LLM call is happening in
        if request.phase:
            phase_name = request.phase.value
            print(f"   Phase: {phase_name} (AgentPhase.{request.phase.name})")

            # Different logging based on phase
            if request.phase == AgentPhase.PLAN:
                print(f"   🎯 Planning phase - Creating execution strategy")
            elif request.phase == AgentPhase.THINK:
                print(f"   🤔 Thinking phase - Selecting next action")
            elif request.phase == AgentPhase.REFLECT:
                print(f"   💭 Reflection phase - Evaluating progress")
        else:
            print(f"   Phase: unknown")

        print(f"   Messages: {len(request.messages)}")

        # ✅ USE AgentRuntime: Access runtime context if available
        if request.runtime:
            runtime = request.runtime
            print(f"   Agent: {runtime.agent_name}")
            print(f"   Step: {runtime.current_step}/{runtime.max_steps}")
            if runtime.run_id:
                print(f"   Run ID: {str(runtime.run_id)[:8]}...")

        # CRITICAL: await the async handler
        response = await handler(request)
        print(f"   Response length: {len(response.content)} chars")
        return response

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolCallResult:
        """Intercept and log tool calls - demonstrates AgentRuntime usage"""
        self.tool_count += 1
        print(f"\n🔧 [Observability] Tool Call #{self.tool_count}: {request.tool_name}")
        print(f"   Arguments: {request.arguments}")

        # ✅ USE AgentRuntime: Access runtime context
        if request.runtime:
            runtime = request.runtime
            print(f"   Agent: {runtime.agent_name}")
            print(f"   Current Phase: {runtime.current_phase.value if runtime.current_phase else 'N/A'}")
            print(f"   Step: {runtime.current_step}/{runtime.max_steps}")

        # CRITICAL: await the async handler
        result = await handler(request)
        print(f"   Success: {result.success}")
        return result

    def on_plan_phase(self, runtime: AgentRuntime, phase_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Track PLAN phase - demonstrates AgentPhase hook"""
        print(f"\n🎯 [Observability] PLAN Phase Started")
        print(f"   Agent: {runtime.agent_name}")
        print(f"   Phase: {AgentPhase.PLAN.value}")

        self.phase_transitions.append({
            "phase": AgentPhase.PLAN.value,
            "agent": runtime.agent_name,
            "step": runtime.current_step
        })
        return None

    def on_reflect_phase(self, runtime: AgentRuntime, phase_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Track REFLECT phase - demonstrates AgentPhase hook"""
        print(f"\n💭 [Observability] REFLECT Phase Started")
        print(f"   Agent: {runtime.agent_name}")
        print(f"   Phase: {AgentPhase.REFLECT.value}")
        print(f"   Current Step: {runtime.current_step}/{runtime.max_steps}")

        self.phase_transitions.append({
            "phase": AgentPhase.REFLECT.value,
            "agent": runtime.agent_name,
            "step": runtime.current_step
        })
        return None

    def on_finish_phase(self, runtime: AgentRuntime, phase_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Track FINISH phase - demonstrates AgentPhase hook"""
        print(f"\n🏁 [Observability] FINISH Phase Started")
        print(f"   Agent: {runtime.agent_name}")
        print(f"   Phase: {AgentPhase.FINISH.value}")
        print(f"   Total Steps: {runtime.current_step}")

        self.phase_transitions.append({
            "phase": AgentPhase.FINISH.value,
            "agent": runtime.agent_name,
            "step": runtime.current_step
        })

        # Print phase transition summary
        if self.phase_transitions:
            print(f"\n   📊 Phase Transition Summary:")
            for i, transition in enumerate(self.phase_transitions, 1):
                print(f"      {i}. {transition['phase']} (step {transition['step']})")

        return None


# ============================================================================
# Example 1: Basic Planning with Middleware
# ============================================================================

async def example_1_basic_planning():
    """Example 1: Basic Plan-Act-Reflect loop"""
    print("\n" + "="*80)
    print("Example 1: Basic Plan-Act-Reflect Loop")
    print("="*80)

    llm = ChatBot(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.3
    )

    agent = ToolCallAgent(
        name="planning-agent",
        llm=llm,
        system_prompt="You are an AI assistant who follows the Plan-Act-Reflect methodology.",

        # Add middleware
        middleware=[
            ObservabilityMiddleware(),          # Observability
            PlanningMiddleware(auto_plan=True), # Auto planning
        ],

        # Enable Plan-Act-Reflect
        enable_plan_phase=True,
        enable_reflect_phase=True,
        reflect_interval=2,  # Reflect every 2 steps

        max_steps=15
    )

    result = await agent.run("Explain the concept of 'Plan-Act-Reflect' in AI agents and how it improves agent reasoning")

    print(f"\n✅ Final Result:\n{result}")

    # View agent state
    diagnostics = agent.get_diagnostics()
    print(f"\n📈 Agent Diagnostics:")
    print(f"   - Execution steps: {diagnostics['current_step']}/{diagnostics['max_steps']}")
    print(f"   - Middleware count: {diagnostics.get('middleware_count', 0)}")
    print(f"   - State keys: {diagnostics.get('agent_state_keys', [])}")


# ============================================================================
# Example 2: HITL (Human-in-the-Loop)
# ============================================================================

async def example_2_hitl():
    """Example 2: Human-in-the-Loop approval flow"""
    print("\n" + "="*80)
    print("Example 2: HITL - Human Approval for Critical Operations")
    print("="*80)

    # Create auto-approval callback (simulating user decision)
    def auto_approver(request):
        print(f"\n❓ [Approval Request] Tool: {request.tool_name}")
        print(f"   Arguments: {request.arguments}")
        print(f"   Decision: Auto-approved (demo)")
        return ApprovalDecision.APPROVE

    llm = ChatBot(provider="openai", model="gpt-4o-mini")

    # Create tool manager with MCP tools
    tool_manager = ToolManager(tools=[])
    tool_manager.add_tool(await create_deepwiki_tool())
    tool_manager.add_tool(await create_memory_tool())

    agent = ToolCallAgent(
        name="hitl-agent",
        llm=llm,
        available_tools=tool_manager,

        # HITL middleware
        middleware=[
            HITLMiddleware(
                interrupt_on={
                    "create_entities": True  # Memory operations require approval
                },
                approval_callback=auto_approver  # Auto-approval (demo)
            )
        ],

        max_steps=3
    )

    result = await agent.run("Analyze the GitHub repository 'XSpoonAi/spoon-core', then store the key insights in memory")

    print(f"\n✅ Final Result:\n{result}")


# ============================================================================
# Example 3: Subagent Orchestration
# ============================================================================

async def example_3_subagents():
    """Example 3: Hierarchical subagent delegation"""
    print("\n" + "="*80)
    print("Example 3: Subagent Orchestration - Specialized Division of Labor")
    print("="*80)

    llm = ChatBot(provider="openai", model="gpt-4o")

    # Create MCP tools first (async)
    deepwiki_tool = await create_deepwiki_tool()
    filesystem_tool = await create_filesystem_tool()

    # Define specialized subagents with MCP tools
    research_agent = SubAgentSpec(
        name="researcher",
        description="Specialized agent for GitHub repository analysis and research",
        system_prompt="""You are a professional GitHub repository analyst.

Your responsibilities:
- Analyze GitHub repositories using DeepWiki
- Gather project insights, architecture, and documentation
- Summarize key findings about repositories

Execution style: Systematic, comprehensive, accurate.""",
        tools=[deepwiki_tool],
        max_steps=5
    )

    analyst_agent = SubAgentSpec(
        name="analyst",
        description="Specialized agent for file and data analysis",
        system_prompt="""You are a data analysis expert.

Your responsibilities:
- Analyze files and data
- Extract key insights and patterns
- Provide actionable recommendations

Execution style: Logically rigorous, deeply insightful.""",
        tools=[filesystem_tool],
        max_steps=5
    )

    # Create orchestrator agent
    agent = ToolCallAgent(
        name="orchestrator",
        llm=llm,
        system_prompt="You are an orchestrator skilled at delegating complex tasks to specialized subagents.",

        # Subagent middleware
        middleware=[
            SubAgentMiddleware(subagents=[research_agent, analyst_agent])
        ],

        max_steps=20
    )

    result = await agent.run("""
    Complete the following tasks:
    1. Analyze the GitHub repository "XSpoonAi/spoon-core" (delegate to researcher)
    2. Extract and summarize the project's architecture and key features (delegate to analyst)

    Please use subagents to complete specialized work.
    """)

    print(f"\n✅ Final Result:\n{result}")


# ============================================================================
# Example 4: Checkpointing and Persistence
# ============================================================================

async def example_4_checkpointing():
    """Example 4: Session persistence and restoration"""
    print("\n" + "="*80)
    print("Example 4: Checkpointing - Session Persistence")
    print("="*80)

    # Create checkpointer
    checkpointer = SQLiteCheckpointer("demo_agent.db")

    llm = ChatBot(provider="openai", model="gpt-4o-mini")

    # First run
    print("\n[First Run] Creating new session...")
    agent1 = ToolCallAgent(
        name="persistent-agent",
        llm=llm,
        thread_id="demo-session-123",  # Session ID

        middleware=[
            CheckpointMiddleware(checkpointer, save_frequency=1)
        ],

        max_steps=3
    )

    result1 = await agent1.run("My name is Alice and I'm studying AI agent development and LLM orchestration patterns")
    print(f"Result 1: {result1}")

    # Second run (restore session)
    print("\n[Second Run] Restoring session...")
    agent2 = ToolCallAgent(
        name="persistent-agent-2",
        llm=llm,
        thread_id="demo-session-123",  # Same session ID

        middleware=[
            CheckpointMiddleware(checkpointer, auto_restore=True)
        ],

        max_steps=3
    )

    result2 = await agent2.run("What is my name and what am I studying?")  # Should remember from session
    print(f"Result 2: {result2}")

    # View checkpoint history
    history = checkpointer.get_history("demo-session-123", limit=5)
    print(f"\n📜 Checkpoint history: {len(history)} records")


# ============================================================================
# Example 5: Full Integration - All Features Combined
# ============================================================================

async def example_5_full_integration():
    """Example 5: Complete Deep Agent - All features integrated"""
    print("\n" + "="*80)
    print("Example 5: Complete Deep Agent System - 6 Core Capabilities")
    print("="*80)

    # Auto-approval callback
    def smart_approver(request):
        # Can make intelligent decisions based on tool and arguments
        print(f"\n🤔 [Smart Approval] Evaluating tool: {request.tool_name}")
        return ApprovalDecision.APPROVE

    # Checkpointer
    checkpointer = SQLiteCheckpointer("full_demo.db")

    # LLM
    llm = ChatBot(provider="openai", model="gpt-4o")

    # Create MCP tools first (async)
    deepwiki_tool = await create_deepwiki_tool()
    filesystem_tool = await create_filesystem_tool()
    memory_tool = await create_memory_tool()

    # Subagents with MCP tools
    subagents = [
        SubAgentSpec(
            name="researcher",
            description="GitHub repository analysis expert",
            system_prompt="You are a research expert focused on analyzing GitHub repositories using DeepWiki.",
            tools=[deepwiki_tool],
            max_steps=5
        ),
        SubAgentSpec(
            name="analyst",
            description="File and data analysis expert",
            system_prompt="You are a data analyst expert focused on extracting insights from files.",
            tools=[filesystem_tool],
            max_steps=5
        )
    ]

    # Create tool manager with MCP tools
    tool_manager = ToolManager(tools=[])
    tool_manager.add_tool(deepwiki_tool)
    tool_manager.add_tool(filesystem_tool)
    tool_manager.add_tool(memory_tool)

    # Assemble complete Deep Agent
    agent = ToolCallAgent(
        name="deep-agent",
        llm=llm,
        thread_id="full-demo-session",
        available_tools=tool_manager,

        # Complete middleware stack
        middleware=[
            ObservabilityMiddleware(),                           # 1. Observability
            CheckpointMiddleware(checkpointer),                  # 2. Persistence
            PlanningMiddleware(auto_plan=True),                  # 3. Planning
            SubAgentMiddleware(subagents=subagents),             # 4. Subagent
            HITLMiddleware(                                      # 5. Human approval
                interrupt_on={"create_entities": True},  # Memory operations need approval
                approval_callback=smart_approver
            ),
        ],

        # Enable all advanced features
        enable_plan_phase=True,       # Plan phase
        enable_reflect_phase=True,    # Reflect phase
        reflect_interval=3,           # Reflect every 3 steps

        max_steps=10
    )

    # Execute complex task
    result = await agent.run("""
    Execute the following complex task:

    1. Analyze the GitHub repository "XSpoonAi/spoon-core" using DeepWiki (can delegate to researcher)
    2. Extract key architectural patterns and features (can delegate to analyst)
    3. Store the findings in memory
    4. Compile a concise report

    Note: This is a demonstration showcasing all Deep Agent capabilities with MCP tools.
    """)

    print(f"\n✅ Final Report:\n{result}")

    # Diagnostic information
    diagnostics = agent.get_diagnostics()
    print(f"\n📊 System Diagnostics:")
    print(f"   - Middleware count: {diagnostics.get('middleware_count', 0)}")
    print(f"   - Execution steps: {diagnostics['current_step']}/{diagnostics['max_steps']} (Note: resets to 0 after run)")
    print(f"   - Internal state: {diagnostics.get('agent_state_keys', [])}")
    print(f"   - Message count: {diagnostics.get('memory_messages', 0)}")

    # Additional diagnostics: check actual agent state
    if hasattr(agent, '_agent_state'):
        print(f"\n🔍 Detailed State Check:")
        print(f"   - _agent_state content: {agent._agent_state}")
        if agent._middleware_pipeline:
            print(f"   - Middleware pipeline exists: ✓")
            # Check planning middleware state
            for mw in agent.middleware:
                mw_name = mw.__class__.__name__
                print(f"   - {mw_name}: ", end="")
                if hasattr(mw, '_current_plan'):
                    if mw._current_plan:
                        print(f"Plan created (goal: {mw._current_plan.goal[:50]}...)")
                    else:
                        print("No plan")
                else:
                    print("Active")
        else:
            print(f"   - Middleware pipeline: ✗")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run all examples"""
    print("\n🚀 Deep Agent Complete Example - spoon-core v2.0\n")
    print("Demonstrating 6 core capabilities:")
    print("1. ✅ Agent Harness: Plan-Act-Reflect loop")
    print("2. ✅ Backends: Unified LLM abstraction (OpenAI/Anthropic/Gemini etc.)")
    print("3. ✅ Subagents: Hierarchical agent delegation")
    print("4. ✅ HITL: Human approval for critical operations")
    print("5. ✅ Memory: Session persistence")
    print("6. ✅ Middleware: Pluggable hook system")
    print("\n" + "="*80 + "\n")

    # Select examples to run (uncomment as needed)

    # Example 1: Basic Planning
    await example_1_basic_planning()

    # Example 2: HITL approval
    await example_2_hitl()

    # Example 3: Subagent orchestration
    await example_3_subagents()

    # Example 4: Checkpointing
    await example_4_checkpointing()

    # Example 5: Full integration (recommended)
    await example_5_full_integration()

    print("\n✅ All examples completed!\n")


if __name__ == "__main__":
    asyncio.run(main())
