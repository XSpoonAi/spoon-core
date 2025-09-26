"""EVM Agent for blockchain operations using SpoonAI framework."""

import logging
from typing import List, Optional, Dict, Any, Union
import asyncio

from pydantic import Field

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.schema import AgentState, Message
from spoon_ai.tools import BaseTool, ToolManager

logger = logging.getLogger(__name__)

def create_configured_chatbot():
    """Create a ChatBot instance with intelligent provider selection."""
    from spoon_ai.llm.config import ConfigurationManager

    try:
        config_manager = ConfigurationManager()
        optimal_provider = config_manager.get_default_provider()
        logger.info(f"Creating ChatBot with optimal provider: {optimal_provider}")
        return ChatBot(llm_provider=optimal_provider)
    except Exception as e:
        logger.error(f"Failed to initialize ChatBot with LLM manager: {e}")
        # Fallback to default
        return ChatBot(llm_provider="openai")

class EvmAgent(ToolCallAgent):
    """
    Advanced EVM agent that can understand natural language commands
    and execute appropriate blockchain operations using EVM tools.

    This agent follows the SpoonAI framework patterns and integrates
    seamlessly with the existing agent ecosystem.
    """

    name: str = "evm_agent"
    description: str = "Intelligent EVM blockchain agent with multi-chain support"

    system_prompt: str = """You are an advanced EVM (Ethereum Virtual Machine) blockchain agent.

Your capabilities include:
- Balance queries for ETH and ERC20 tokens
- Native ETH transfers across multiple networks
- ERC20 token transfers (USDC, USDT, DAI, etc.)
- Token swaps via DEX aggregators (1inch, Uniswap, etc.)
- Cross-chain bridging operations via LiFi protocol
- Real-time price quotes and swap estimates
- Multi-network support (Ethereum, Base, Polygon, Arbitrum, Optimism, etc.)

You understand commands in both English and Chinese. Always prioritize user safety by:
1. Clearly explaining what operations will be performed
2. Validating addresses, amounts, and network parameters
3. Providing detailed feedback with transaction information
4. Using appropriate error handling and confirmations

Parse user instructions intelligently and execute the most appropriate EVM operation using the available tools.
When working with blockchain operations, be precise about amounts, addresses, and network selections.
    """.strip()

    next_step_prompt: str = "Analyze the blockchain request and determine the appropriate EVM operation to execute. Use the available tools to complete the task safely and efficiently."

    max_steps: int = 15
    tool_choice: str = "auto"

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    llm: ChatBot = Field(default_factory=create_configured_chatbot)

    # EVM-specific configuration
    default_network: str = Field(default="polygon")
    confirmation_required: bool = Field(default=True)
    gas_estimation_enabled: bool = Field(default=True)

    def __init__(self, **kwargs):
        """Initialize EVM Agent with tools and configuration."""
        # Initialize parent ToolCallAgent
        super().__init__(**kwargs)

        # Load EVM tools
        self._load_evm_tools()

        logger.info(f"Initialized EVM Agent: {self.name} with {len(self.avaliable_tools.tool_map)} tools")

    def _load_evm_tools(self):
        """Load all available EVM tools into the agent."""
        try:
            from spoon_toolkits.crypto.evm import (
                EvmBalanceTool,
                EvmTransferTool,
                EvmErc20TransferTool,
                EvmSwapTool,
                EvmBridgeTool,
                EvmSwapQuoteTool
            )

            # Create tool instances
            evm_tools = [
                EvmBalanceTool(),
                EvmTransferTool(),
                EvmErc20TransferTool(),
                EvmSwapTool(),
                EvmBridgeTool(),
                EvmSwapQuoteTool(),
            ]

            # Update tool manager
            self.avaliable_tools = ToolManager(evm_tools)

            logger.info(f"Loaded {len(evm_tools)} EVM tools: {[tool.name for tool in evm_tools]}")

        except ImportError as e:
            logger.error(f"Failed to import EVM tools: {e}")
            logger.warning("EVM Agent will operate without EVM tools")
        except Exception as e:
            logger.error(f"Failed to load EVM tools: {e}")
            raise

    async def initialize(self, __context: Any = None):
        """Initialize async components."""
        logger.info(f"Initializing EVM Agent '{self.name}'")

        try:
            # Test LLM connection
            connection_test = await self._test_llm_connection()
            if connection_test["status"] == "success":
                logger.info(" EVM Agent LLM connection successful")
            else:
                logger.warning(f" EVM Agent LLM connection failed: {connection_test.get('error', 'Unknown error')}")

            # Validate tools
            available_tools = list(self.avaliable_tools.tool_map.keys())
            logger.info(f" EVM Agent available tools: {available_tools}")

            if not available_tools:
                logger.warning(" No tools loaded - agent functionality will be limited")

        except Exception as e:
            logger.error(f"Failed to initialize EVM Agent {self.name}: {str(e)}")
            if __context and hasattr(__context, 'report_error'):
                await __context.report_error(e)
            raise

        logger.info(f" EVM Agent '{self.name}' initialized successfully")

    async def _test_llm_connection(self) -> Dict[str, Any]:
        """Test the LLM connection."""
        try:
            test_message = Message(role="user", content="Hello, please respond with 'Connection successful'")
            response = await self.llm.ask([test_message])
            return {
                "status": "success",
                "response": response
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def process_evm_command(
        self,
        command: str,
        network: Optional[str] = None,
        private_key: Optional[str] = None,
        confirm: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a natural language EVM command.

        Args:
            command: Natural language blockchain command
            network: Target network (ethereum, polygon, base, etc.)
            private_key: Private key for transactions
            confirm: Whether to require confirmation
            **kwargs: Additional parameters

        Returns:
            Dict containing execution results
        """
        # Use defaults if not specified
        network = network or self.default_network
        confirm = confirm if confirm is not None else self.confirmation_required

        # Build enhanced instruction
        enhanced_command = f"""
Execute this EVM blockchain operation: {command}

Parameters:
- Network: {network}
- Confirmation required: {confirm}
- Private key provided: {'Yes' if private_key else 'No'}

Please analyze the command and use the appropriate EVM tools to execute it safely.
        """.strip()

        try:
            # Run the command using the agent's run method
            result = await self.run(enhanced_command)

            return {
                "success": True,
                "result": result,
                "command": command,
                "network": network,
                "parameters": kwargs
            }

        except Exception as e:
            logger.error(f"EVM command execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "network": network
            }

    def get_supported_networks(self) -> List[Dict[str, Any]]:
        """Get list of supported EVM networks."""
        return [
            {
                "name": "Ethereum Mainnet",
                "id": "ethereum",
                "chain_id": 1,
                "native_token": "ETH",
                "gas_level": "High"
            },
            {
                "name": "Polygon",
                "id": "polygon",
                "chain_id": 137,
                "native_token": "MATIC",
                "gas_level": "Very Low"
            },
            {
                "name": "Base",
                "id": "base",
                "chain_id": 8453,
                "native_token": "ETH",
                "gas_level": "Low"
            },
            {
                "name": "Arbitrum One",
                "id": "arbitrum",
                "chain_id": 42161,
                "native_token": "ETH",
                "gas_level": "Low"
            },
            {
                "name": "Optimism",
                "id": "optimism",
                "chain_id": 10,
                "native_token": "ETH",
                "gas_level": "Low"
            },
            {
                "name": "Ethereum Sepolia Testnet",
                "id": "sepolia",
                "chain_id": 11155111,
                "native_token": "ETH",
                "gas_level": "Free (Testnet)",
                "is_testnet": True
            }
        ]

    def get_supported_operations(self) -> List[Dict[str, Any]]:
        """Get list of supported EVM operations."""
        return [
            {
                "operation": "balance_query",
                "description": "Check ETH or token balance for an address",
                "examples": [
                    "Check my ETH balance",
                    "查看我的USDC余额",
                    "What's the balance of 0x123... on Polygon?"
                ]
            },
            {
                "operation": "eth_transfer",
                "description": "Send native ETH to another address",
                "examples": [
                    "Send 0.1 ETH to 0x456...",
                    "转账0.05个ETH给朋友",
                    "Transfer ETH to my other wallet"
                ]
            },
            {
                "operation": "token_transfer",
                "description": "Send ERC20 tokens to another address",
                "examples": [
                    "Send 100 USDC to 0x789...",
                    "转移50个USDT到我的钱包",
                    "Transfer DAI tokens"
                ]
            },
            {
                "operation": "token_swap",
                "description": "Exchange tokens via DEX aggregators",
                "examples": [
                    "Swap 1 ETH for USDC",
                    "用100 USDC买ETH",
                    "Exchange tokens on Uniswap"
                ]
            },
            {
                "operation": "cross_chain_bridge",
                "description": "Move tokens between blockchains",
                "examples": [
                    "Bridge 50 USDC to Base",
                    "跨链转移代币到Polygon",
                    "Move ETH to Arbitrum"
                ]
            },
            {
                "operation": "price_quote",
                "description": "Get real-time swap quotes",
                "examples": [
                    "Get quote for 1 ETH to USDC",
                    "获取价格报价",
                    "How much USDT for 100 USDC?"
                ]
            }
        ]

    def add_evm_tool(self, tool: BaseTool) -> None:
        """Add a new EVM tool to the agent."""
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Tool must be an instance of BaseTool, got {type(tool)}")

        if not hasattr(tool, 'name') or not tool.name:
            raise ValueError("Tool must have a valid name")

        # Check for duplicates
        if tool.name in self.avaliable_tools.tool_map:
            logger.warning(f"Tool '{tool.name}' already exists, replacing it")

        # Add to tool manager
        self.avaliable_tools.tool_map[tool.name] = tool
        if not hasattr(self.avaliable_tools, 'tools'):
            self.avaliable_tools.tools = list(self.avaliable_tools.tool_map.values())
        else:
            self.avaliable_tools.tools.append(tool)

        logger.info(f"Added EVM tool: {tool.name}")

    def get_tool_help(self) -> str:
        """Generate help text for available EVM operations."""
        help_text = """
 EVM Agent - Blockchain Operations Help

SUPPORTED NETWORKS:
"""
        for network in self.get_supported_networks():
            testnet_flag = " (TESTNET)" if network.get("is_testnet") else ""
            help_text += f"• {network['name']}{testnet_flag} - Gas: {network['gas_level']}\n"

        help_text += """
AVAILABLE OPERATIONS:

 Balance Queries:
• "Check my ETH balance"
• "查看我的USDC余额"
• "What's my balance on Base?"

 Transfers:
• "Send 0.1 ETH to 0x..."
• "转账50个USDC给朋友"
• "Transfer 100 USDT to my wallet"

 Token Swaps:
• "Swap 1 ETH for USDC"
• "用100 USDC换ETH"
• "Exchange tokens on Uniswap"

 Cross-chain Bridges:
• "Bridge 100 USDC to Base"
• "跨链转移50个USDT到Polygon"
• "Move ETH to Arbitrum"

 Price Quotes:
• "Get quote for 1 ETH to USDC"
• "获取价格报价"
• "How much can I get for 100 USDC?"

USAGE TIPS:
• Specify amounts and addresses clearly
• I'll ask for confirmation before transactions
• Supports both English and Chinese commands
• Set EVM_PRIVATE_KEY environment variable for transactions

SAFETY FEATURES:
• Transaction confirmation prompts
• Network and address validation
• Balance verification before transfers
• Detailed transaction feedback with explorer links
        """
        return help_text.strip()