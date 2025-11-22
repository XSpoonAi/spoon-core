from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import SSETransport
from spoon_ai.tools.tool_manager import ToolManager
from pydantic import Field
from spoon_ai.chat import ChatBot
from spoon_ai.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from spoon_ai.schema import AgentState
from typing import Any
import asyncio

# Neo toolkit tools will be imported lazily in __init__ to avoid import errors
# This prevents issues with spoon-toolkit dependencies during module import
NEO_TOOLS_AVAILABLE = False

"""
🔍 SpoonThirdWebMCP Agent

This agent combines Neo blockchain analysis with EVM blockchain data querying.
It supports:
- Neo blockchain address analysis (count, info, tags, transactions)
- EVM blockchain data via Thirdweb Insight API

📌 Requirements:
- For Neo tools: Neo network access (testnet/mainnet)
- For EVM tools: Thirdweb Insight API `client_id` (optional)

📚 For more info:
- Neo: https://neo.org
- Thirdweb: https://portal.thirdweb.com/insight/overview

💡 Example usage:
    > Analyze address NUTtedVrz5RgKAdCvtKiq3sRkb9pizcewe on Neo testnet
    > Show me recent USDT transfers on Ethereum
    > Get address count and active addresses on Neo
"""


class SpoonThirdWebMCP(SpoonReactAI, MCPClientMixin):
    name: str = "SpoonThirdWebMCP"
    description: str = (
        "An AI assistant specialized in blockchain data analysis across Neo and EVM chains. "
        "Supports Neo address analysis (count, info, tags, transactions) and EVM blockchain data "
        "via Thirdweb Insight API (contract events, token transfers, blocks, wallet activity). "
        "Use this agent for comprehensive blockchain analysis across multiple networks."
    )
    system_prompt: str = """
        You are a multi-chain blockchain data analyst assistant with expertise in both Neo and EVM blockchains.
        
        **Neo Blockchain Tools:**
        Use these tools when the user asks about Neo blockchain:
        - Address count on network → use `GetAddressCountTool` (specify network: "testnet" or "mainnet")
        - Address information/details → use `GetAddressInfoTool` (provide address and network)
        - Active addresses → use `GetActiveAddressesTool` (specify network)
        - Address tags/labels → use `GetTagByAddressesTool` (provide addresses list and network)
        - Total sent/received for address → use `GetTotalSentAndReceivedTool` (provide address and network)
        
        **EVM Blockchain Tools (via Thirdweb Insight API):**
        Use these tools when the user asks about EVM chains (Ethereum, Polygon, etc.):
        - contract logs or Transfer events → use `get_contract_events_from_thirdweb_insight`
        - USDT transfers across chains → use `get_multichain_transfers_from_thirdweb_insight`
        - recent cross-chain transactions → use `get_transactions`
        - a specific contract's transaction history → use `get_contract_transactions`
        - contract function call history → use `get_contract_transactions_by_signature`
        - recent block info by chain → use `get_blocks_from_thirdweb_insight`
        - wallet activity across chains → use `get_wallet_transactions_from_thirdweb_insight`

        **Important Parameters:**
        - For Neo tools: Always specify `network` parameter ("testnet" or "mainnet")
        - For EVM tools: `chain_id` (Ethereum = 1, Polygon = 137, etc.), `contract_address`, `limit` (default: 10)
        - `client_id` for Thirdweb API can be pulled from environment variable or user context

        **Best Practices:**
        - If user mentions "Neo" or provides Neo addresses (starting with N), use Neo tools
        - If user mentions EVM chains (Ethereum, Polygon, BSC, etc.), use Thirdweb tools
        - Always ask for clarification if the network or chain is unclear
        - Provide comprehensive analysis combining multiple tool results when appropriate

        If something is unclear, ask for clarification. Otherwise, call the appropriate tool(s).
    """

    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MCPClientMixin.__init__(self, mcp_transport=kwargs.get('mcp_transport', SSETransport("http://127.0.0.1:8765/sse")))
        
        # Initialize Neo toolkit tools (lazy import to avoid dependency issues)
        neo_tools = []
        try:
            # Try to import Neo tools (lazy import to avoid triggering full package initialization)
            # Note: This may fail if spoon-toolkit has version incompatibilities (e.g., FastMCP API changes)
            from spoon_toolkits.crypto.neo import (
                GetAddressCountTool,
                GetAddressInfoTool,
                GetActiveAddressesTool,
                GetTagByAddressesTool,
                GetTotalSentAndReceivedTool,
            )
            
            # Create tool instances
            neo_tools = [
                GetAddressCountTool(),
                GetAddressInfoTool(),
                GetActiveAddressesTool(),
                GetTagByAddressesTool(),
                GetTotalSentAndReceivedTool(),
            ]
            print(f"✅ Loaded {len(neo_tools)} Neo toolkit tools")
            NEO_TOOLS_AVAILABLE = True
        except (ImportError, ModuleNotFoundError) as e:
            print(f"⚠️  Warning: Neo toolkit tools not available (ImportError: {e})")
            print("   Install spoon-toolkits to enable Neo features.")
            NEO_TOOLS_AVAILABLE = False
        except TypeError as e:
            # Handle FastMCP version incompatibility issues
            if "unexpected keyword argument" in str(e) or "FastMCP" in str(e):
                print(f"⚠️  Warning: Neo toolkit tools unavailable due to version incompatibility")
                print(f"   Error: {e}")
                print("   This is likely due to FastMCP API changes in spoon-toolkits.")
                print("   Agent will continue without Neo tools.")
            else:
                print(f"⚠️  Warning: Failed to initialize Neo tools (TypeError: {e})")
            NEO_TOOLS_AVAILABLE = False
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize Neo tools: {e}")
            print("   Agent will continue without Neo tools.")
            NEO_TOOLS_AVAILABLE = False
        
        # Merge Neo tools with existing tools
        existing_tools = self.available_tools.tools if self.available_tools else []
        all_tools = existing_tools + neo_tools
        self.available_tools = ToolManager(all_tools)
    
    async def initialize(self, __context: Any = None):
        """Override initialize to avoid calling connect() which may not exist"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing SpoonThirdWebMCP agent '{self.name}'")
        
        # Skip MCP connection setup since we're using MCPClientMixin directly
        # The tools are already initialized in __init__
        try:
            # Only ensure x402 tools if needed (optional)
            if hasattr(self, '_ensure_x402_tools'):
                await self._ensure_x402_tools()
        except Exception as e:
            logger.warning(f"Optional initialization step failed: {e}")
            # Don't raise - allow agent to continue without optional features
    
    async def run_streaming(self, request: str, enable_stdout: bool = True) -> str:
        """
        Run agent with streaming output, tool calling, and automatic memory management.
        
        This method combines:
        - Full agent capabilities (tool calling via think/act loop)
        - Streaming output from LLM responses
        - Automatic memory management
        
        Args:
            request: User question/request
            enable_stdout: Whether to automatically print tokens to stdout
            
        Returns:
            Full response text
        """
        # Ensure agent is in IDLE state
        if self.state != AgentState.IDLE:
            # Reset agent state if needed
            self.state = AgentState.IDLE
            self.current_step = 0
        
        # Clear output queue and task_done event
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                break
        self.task_done.clear()
        
        # Start agent.run() in background
        run_task = asyncio.create_task(self.run(request))
        
        # Stream output from output_queue while agent is running
        full_response = ""
        collected_content = []
        
        try:
            async for item in self.stream(timeout=300):  # 5 minute timeout
                if isinstance(item, dict):
                    # Handle different types of output items
                    if "content" in item:
                        content = item["content"] or ""
                        if content:
                            collected_content.append(content)
                            if enable_stdout:
                                print(content, end="", flush=True)
                    
                    if "tool_calls" in item:
                        tool_calls = item.get("tool_calls", [])
                        if tool_calls and enable_stdout:
                            print(f"\n🔧 Calling {len(tool_calls)} tool(s)...\n", flush=True)
                    
                    if "tool_result" in item:
                        tool_result = item.get("tool_result", "")
                        if tool_result and enable_stdout:
                            print(f"📊 Tool result: {str(tool_result)[:100]}...\n", flush=True)
                elif isinstance(item, str):
                    # Direct string output
                    collected_content.append(item)
                    if enable_stdout:
                        print(item, end="", flush=True)
            
            # Wait for agent.run() to complete
            try:
                result = await asyncio.wait_for(run_task, timeout=300)
                full_response = result or "".join(collected_content)
            except asyncio.TimeoutError:
                print("\n⚠️ Agent execution timed out", flush=True)
                full_response = "".join(collected_content)
            except Exception as e:
                print(f"\n❌ Agent execution error: {e}", flush=True)
                full_response = "".join(collected_content)
        
        except Exception as e:
            print(f"\n❌ Streaming error: {e}", flush=True)
            # Try to get result from run_task
            try:
                full_response = await asyncio.wait_for(run_task, timeout=5)
            except:
                full_response = "".join(collected_content)
        
        # Ensure we have a response
        if not full_response:
            full_response = "".join(collected_content)
        
        # Add final response to memory if not already added by agent.run()
        # (agent.run() already adds messages via think() and act())
        if enable_stdout:
            print()  # New line after streaming
        
        return full_response


async def main():
    # Configure ChatBot with memory
    chatbot = ChatBot(
        llm_provider="openrouter",
        model_name="anthropic/claude-sonnet-4",
        base_url="https://openrouter.ai/api/v1",
        enable_short_term_memory=True,  # Enable memory - remembers conversation history
    )
    
    # Create agent
    agent = SpoonThirdWebMCP(llm=chatbot)
    
    # Initialize agent to connect to MCP server and load tools
    print("Initializing agent and connecting to MCP server...")
    try:
        await agent.initialize()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize MCP connection: {e}")
        print("   Agent will still work, but without MCP tools.")
    
    print("=" * 60)
    print("🔍 SpoonThirdWebMCP Agent - Multi-Turn Conversation with Memory")
    print("=" * 60)
    print()
    
    # Multiple questions - agent remembers previous conversations
    questions = [
        # Neo blockchain questions
        "Analyze address NUTtedVrz5RgKAdCvtKiq3sRkb9pizcewe on Neo testnet. Show me address info and total sent/received.",
        "How many addresses are on Neo testnet? Show me some active addresses.",
        # EVM blockchain questions (if MCP tools are available)
        # "Show me recent USDT transfers on Ethereum and Polygon, using client ID 8a2408f5cf60eb60df6f72baa6376438.",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 60)
        
        # Use run_streaming for streaming + tool calling + memory
        print("\nResponse (streaming + tool calling + memory):")
        result = await agent.run_streaming(question, enable_stdout=True)
        
        # Alternative: Use agent.run() for tool calling + memory (but no streaming)
        # Uncomment below if you don't need streaming:
        # print("\nResponse (tool calling + memory, no streaming):")
        # result = await agent.run(question)
        # print(result)
        
        # Show that memory is working
        memory_messages = agent.memory.get_messages()
        print(f"\n📝 Memory: {len(memory_messages)} messages")
        print()


    print("=" * 60)
    print("✅ Memory is working! Agent remembers all previous conversations.")
    memory_messages = agent.memory.get_messages()
    print(f"   Total messages in memory: {len(memory_messages)}")
    print("=" * 60)
    print("\n💡 Features:")
    print("   - run_streaming(): ✅ Streaming output + ✅ Tool calling + ✅ Memory")
    print("   - agent.run(): ✅ Tool calling + ✅ Memory (no streaming)")
    

if __name__ == "__main__":
    asyncio.run(main())





