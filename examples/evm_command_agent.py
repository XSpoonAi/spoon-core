import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

# Ensure toolkit is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../spoon-toolkit')))

from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.chat import ChatBot
from spoon_toolkits.crypto.evm import EvmAgent

logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

# Suppress verbose logs from specific modules
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('spoon_ai').setLevel(logging.WARNING)
logging.getLogger('spoon_toolkits').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.ERROR)

@dataclass
class NetworkConfig:
    """Network configuration for EVM chains."""
    name: str
    chain_id: int
    rpc_url: str
    native_token: str
    explorer_url: str
    gas_level: str
    is_testnet: bool = False

class SpoonEVMCommandAgent(SpoonReactMCP):
    """
    Advanced EVM command parser and execution agent.

    This agent integrates natural language processing with comprehensive EVM operations,
    providing a user-friendly interface for blockchain interactions across multiple networks.
    """

    name: str = "SpoonEVMCommandAgent"
    system_prompt: str = """
You are SpoonOS EVM Command Agent, an intelligent blockchain assistant that can understand
natural language commands and execute various EVM operations safely and efficiently.

Your capabilities include:
- Balance queries for ETH and ERC20 tokens
- Native ETH and ERC20 token transfers
- Token swaps via DEX aggregators
- Cross-chain bridging operations
- Real-time price quotes and analysis
- Multi-network support (Ethereum, Base, Polygon, etc.)
- Safety confirmations and transaction monitoring

You support both English and Chinese commands and always prioritize user safety by:
1. Clearly explaining what operations will be performed
2. Requesting confirmation for transactions
3. Providing detailed feedback with transaction hashes
4. Offering help and examples when commands are unclear

Parse user commands intelligently and execute the appropriate EVM operations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize EVM agent
        self.evm_agent = EvmAgent()

        # Configure supported networks
        self.supported_networks = {
            "ethereum": NetworkConfig(
                name="Ethereum Mainnet",
                chain_id=1,
                rpc_url=os.getenv("ETHEREUM_RPC_URL", "https://eth-mainnet.alchemyapi.io/v2/demo"),
                native_token="ETH",
                explorer_url="https://etherscan.io",
                gas_level="High"
            ),
            "base": NetworkConfig(
                name="Base",
                chain_id=8453,
                rpc_url=os.getenv("BASE_RPC_URL", "https://mainnet.base.org"),
                native_token="ETH",
                explorer_url="https://basescan.org",
                gas_level="Low"
            ),
            "polygon": NetworkConfig(
                name="Polygon",
                chain_id=137,
                rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com/"),
                native_token="MATIC",
                explorer_url="https://polygonscan.com",
                gas_level="Very Low"
            ),
            "arbitrum": NetworkConfig(
                name="Arbitrum One",
                chain_id=42161,
                rpc_url=os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"),
                native_token="ETH",
                explorer_url="https://arbiscan.io",
                gas_level="Low"
            ),
            "optimism": NetworkConfig(
                name="Optimism",
                chain_id=10,
                rpc_url=os.getenv("OPTIMISM_RPC_URL", "https://mainnet.optimism.io"),
                native_token="ETH",
                explorer_url="https://optimistic.etherscan.io",
                gas_level="Low"
            ),
            "sepolia": NetworkConfig(
                name="Ethereum Sepolia Testnet",
                chain_id=11155111,
                rpc_url=os.getenv("SEPOLIA_RPC_URL", "https://sepolia.gateway.tenderly.co"),
                native_token="ETH",
                explorer_url="https://sepolia.etherscan.io",
                gas_level="Free (Testnet)",
                is_testnet=True
            )
        }

        # Default configuration
        self.default_network = "polygon"  
        self.confirmation_required = True
        self.debug_mode = False

        # Demo/Test addresses for examples (real addresses provided by user)
        self.demo_addresses = {
            "primary_wallet": "0xb018A8428bB87b15401dEcb615895Db2361e81a4",
            "secondary_wallet": "0xFa4aea1F101cbCae0d1e7CE005f2e3DaC25A4b4D",
        }

        # Test private key for secondary wallet (provided by user for testing)
        self.demo_private_key = "0x8eaa2c814a045ce50c2dde8e4eb7b9b5201a63a6e95b91f74f98fad5a403a9f2"

        # Available tools - use the EVM agent as our primary tool
        self.available_tools = ToolManager([self.evm_agent])

    async def connect(self):
        """Connect method required by SpoonReactMCP base class."""
        logger.info("Connecting SpoonOS EVM Command Agent...")
        return True

    async def initialize(self):
        """Initialize the agent and EVM tools."""
        logger.info("Initializing SpoonOS EVM Command Agent...")

        # Initialize parent class
        await super().initialize()

        # Test EVM agent connectivity
        try:
            llm_test = await self.evm_agent.test_llm_connection()
            if llm_test["status"] == "success":
                logger.info(" EVM Agent LLM connection successful")
            else:
                logger.warning(f" EVM Agent LLM connection failed: {llm_test.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f" Failed to test EVM Agent connection: {e}")

        logger.info("SpoonOS EVM Command Agent initialized successfully")

    async def process_command(
        self,
        command: str,
        network: Optional[str] = None,
        private_key: Optional[str] = None,
        confirm: Optional[bool] = None,
        debug: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a natural language command and execute the appropriate EVM operation.

        Args:
            command: Natural language command (English or Chinese)
            network: Target network name (default: base)
            private_key: Private key for transactions (optional, uses env var)
            confirm: Whether to confirm transactions (default: True)
            debug: Enable debug mode (default: False)
            **kwargs: Additional parameters

        Returns:
            Dict containing execution results and metadata
        """
        start_time = datetime.now()

        # Configure parameters
        network = network or self.default_network
        confirm = confirm if confirm is not None else self.confirmation_required
        debug = debug if debug is not None else self.debug_mode

        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"Processing command: '{command}' on network: {network}")

        # Validate network
        if network not in self.supported_networks:
            return {
                "success": False,
                "error": f"Unsupported network '{network}'. Available: {list(self.supported_networks.keys())}",
                "command": command,
                "timestamp": start_time.isoformat()
            }

        network_config = self.supported_networks[network]

        # Get RPC URL and private key
        rpc_url = network_config.rpc_url
        private_key = private_key or os.getenv("EVM_PRIVATE_KEY")

        try:
            # First analyze the command to understand what we're doing
            if debug:
                logger.debug("Analyzing command with EVM agent...")
                analysis = await self.evm_agent.analyze_instruction(command)
                logger.debug(f"Command analysis: {json.dumps(analysis, indent=2, default=str)}")

            # Execute the command using EVM agent
            result = await self.evm_agent.execute(
                instruction=command,
                rpc_url=rpc_url,
                private_key=private_key,
                confirm=confirm,
                debug=debug,
                **kwargs
            )

            # Process the result
            execution_time = (datetime.now() - start_time).total_seconds()

            if result.error:
                return {
                    "success": False,
                    "error": result.error,
                    "command": command,
                    "network": network_config.name,
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time.isoformat()
                }
            else:
                # Extract transaction hash if present
                tx_hash = None
                explorer_link = None

                if hasattr(result.output, 'get') and isinstance(result.output, dict):
                    if 'transaction' in result.output and isinstance(result.output['transaction'], dict):
                        tx_hash = result.output['transaction'].get('hash')
                    elif 'raw_data' in result.output and isinstance(result.output['raw_data'], dict):
                        tx_hash = result.output['raw_data'].get('hash')

                # Generate explorer link if we have a transaction hash
                if tx_hash:
                    explorer_link = f"{network_config.explorer_url}/tx/{tx_hash}"

                return {
                    "success": True,
                    "result": result.output,
                    "command": command,
                    "network": network_config.name,
                    "transaction_hash": tx_hash,
                    "explorer_link": explorer_link,
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time.isoformat()
                }

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}",
                "command": command,
                "network": network_config.name,
                "execution_time_seconds": execution_time,
                "timestamp": start_time.isoformat()
            }

    async def get_network_status(self, network: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for a specific network or all networks."""
        if network:
            if network not in self.supported_networks:
                return {"error": f"Unknown network: {network}"}

            config = self.supported_networks[network]
            try:
                # Test connectivity
                from web3 import Web3, HTTPProvider
                w3 = Web3(HTTPProvider(config.rpc_url))
                is_connected = w3.is_connected()

                status = {
                    "network": config.name,
                    "chain_id": config.chain_id,
                    "connected": is_connected,
                    "native_token": config.native_token,
                    "gas_level": config.gas_level,
                    "is_testnet": config.is_testnet
                }

                if is_connected:
                    try:
                        latest_block = w3.eth.block_number
                        status["latest_block"] = latest_block
                    except:
                        pass

                return status
            except Exception as e:
                return {
                    "network": config.name,
                    "connected": False,
                    "error": str(e)
                }
        else:
            # Return status for all networks
            all_status = {}
            for net_name in self.supported_networks:
                all_status[net_name] = await self.get_network_status(net_name)
            return all_status

    async def list_supported_operations(self) -> List[Dict[str, str]]:
        """List all supported operations with examples."""
        return [
            {
                "operation": "Balance Query",
                "description": "Check ETH or token balance for an address",
                "examples": [
                    "Check my ETH balance",
                    "查看我的USDC余额",
                    "What's my balance on Base?",
                    "Show balance for 0x..."
                ]
            },
            {
                "operation": "ETH Transfer",
                "description": "Send native ETH to another address",
                "examples": [
                    "Send 0.1 ETH to 0x...",
                    "转账0.01个ETH给朋友",
                    "Transfer some ETH to my wallet"
                ]
            },
            {
                "operation": "Token Transfer",
                "description": "Send ERC20 tokens to another address",
                "examples": [
                    "Send 100 USDC to 0x...",
                    "转移50个USDT",
                    "Transfer tokens to address"
                ]
            },
            {
                "operation": "Token Swap",
                "description": "Exchange one token for another via DEX",
                "examples": [
                    "Swap 1 ETH for USDC",
                    "用100 USDC换ETH",
                    "Exchange tokens",
                    "Buy some ETH with USDT"
                ]
            },
            {
                "operation": "Cross-chain Bridge",
                "description": "Move tokens between different blockchains",
                "examples": [
                    "Bridge 50 USDC to Base",
                    "跨链转移代币到Polygon",
                    "Move ETH to Arbitrum"
                ]
            },
            {
                "operation": "Price Quote",
                "description": "Get real-time swap quotes and prices",
                "examples": [
                    "Get quote for 1 ETH to USDC",
                    "获取价格报价",
                    "How much USDT can I get for 100 USDC?",
                    "What's the best price for swapping?"
                ]
            }
        ]

    def get_help_text(self) -> str:
        """Generate comprehensive help text for the agent."""
        help_text = """
🤖 SpoonOS EVM Command Agent Help

OVERVIEW:
I can understand natural language commands and execute various blockchain operations
across multiple EVM networks. I support both English and Chinese commands.

SUPPORTED NETWORKS:
"""
        for key, config in self.supported_networks.items():
            testnet_flag = " (TESTNET)" if config.is_testnet else ""
            help_text += f"• {config.name}{testnet_flag} - Gas: {config.gas_level}\n"

        help_text += """
COMMAND EXAMPLES:

 Balance Queries:
• "Check my ETH balance"
• "查看我的USDC余额"
• "What's my balance on Base?"
• "Show USDT balance for 0x1234..."

 Transfers:
• "Send 0.1 ETH to 0x..."
• "转账50个USDC给朋友"
• "Transfer 100 USDT to my wallet"

 Token Swaps:
• "Swap 1 ETH for USDC"
• "用100 USDC换ETH"
• "Buy some tokens with ETH"
• "Exchange 50 USDT for ETH"

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
• I'll ask for confirmation before executing transactions
• Use network names like 'base', 'polygon', 'ethereum'
• Check your private key environment variable is set
• I support both English and Chinese commands

SAFETY FEATURES:
• Transaction confirmation prompts
• Network validation
• Address format checking
• Balance verification before transfers
• Detailed transaction feedback with explorer links

Need help with a specific command? Just ask!
        """
        return help_text.strip()

async def run_interactive_demo():
    """Run an interactive demonstration of the EVM Command Agent."""
    print("🎬 Starting SpoonOS EVM Command Agent Interactive Demo...")

    # Initialize the agent
    agent = SpoonEVMCommandAgent(llm=ChatBot(llm_provider="openai"))

    try:
        await agent.initialize()

        # Comprehensive demo commands to test all EVM toolkit functions
        demo_commands = [
            # ===== BALANCE QUERIES =====
            {
                "command": f"Check ETH balance for {agent.demo_addresses['primary_wallet']}",
                "network": "polygon",
                "description": " EvmBalanceTool - ETH balance query"
            },
            {
                "command": f"查看地址 {agent.demo_addresses['secondary_wallet']} 的USDC余额",
                "network": "polygon",
                "description": " EvmBalanceTool - Chinese USDC balance query"
            },

            # ===== SWAP QUOTES =====
            {
                "command": "Get quote for swapping 0.01 ETH to USDC",
                "network": "polygon",
                "description": "💱 EvmSwapQuoteTool - ETH to USDC price quote"
            },

            # ===== NATIVE ETH TRANSFERS =====
            {
                "command": f"Send 0.0001 ETH to {agent.demo_addresses['primary_wallet']}",
                "network": "polygon",
                "description": " EvmTransferTool - Native ETH transfer (REAL TX)"
            },

            # ===== ERC20 TOKEN TRANSFERS =====
            {
                "command": f"Send 0.01 USDC to {agent.demo_addresses['primary_wallet']}",
                "network": "polygon",
                "description": " EvmErc20TransferTool - USDC transfer (REAL TX)"
            },

            # ===== TOKEN SWAPS =====
            {
                "command": "Swap 0.001 ETH for USDC",
                "network": "polygon",
                "description": " EvmSwapTool - ETH to USDC swap (REAL TX)"
            },

            # ===== CROSS-CHAIN BRIDGES =====
            {
                "command": "Bridge 0.01 USDC to Ethereum",
                "network": "polygon",
                "description": " EvmBridgeTool - USDC bridge to Ethereum "
            },
        ]

        print(f"\n🎭 Running Comprehensive EVM Toolkit Demo ({len(demo_commands)} operations)...")
        print("🚀 Testing all EVM functions: Balance, Quote, Transfer, Swap, Bridge")

        for i, demo in enumerate(demo_commands, 1):
            print(f"\n{'='*60}")
            print(f"Demo {i}/{len(demo_commands)}: {demo['description']}")
            print(f"Command: \"{demo['command']}\"")
            print(f"Network: {demo['network']}")
            print(f"{'='*60}")

            try:
                result = await agent.process_command(
                    command=demo['command'],
                    network=demo['network'],
                    private_key=agent.demo_private_key,  # Use demo private key
                    confirm=False,  # Skip confirmations in demo
                    debug=False  # Disable debug to reduce log noise
                )

                print(f"\n Execution Result:")
                if result['success']:
                    print(f"    ✅ Status: SUCCESS")
                    print(f"     Execution Time: {result.get('execution_time_seconds', 0):.2f}s")

                    # Show result details
                    if 'result' in result:
                        result_data = result['result']
                        if isinstance(result_data, dict) and 'message' in result_data:
                            print(f"    Result: {result_data['message']}")
                        else:
                            print(f"    Result: {result_data}")

                    # Show transaction info if available
                    if result.get('transaction_hash'):
                        print(f"    Transaction: {result['transaction_hash']}")
                        print(f"    Explorer: {result.get('explorer_link', 'N/A')}")

                else:
                    print(f"    Status: FAILED")
                    print(f"    Error: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"     Demo failed: {e}")

            # Small delay for readability
            await asyncio.sleep(2)


    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()

async def run_interactive_session():
    """Run an interactive session where users can input commands."""
    agent = SpoonEVMCommandAgent(llm=ChatBot(llm_provider="openai"))

    try:
        await agent.initialize()

        print(f"\nInteractive Mode Started")
        print(" Try commands like:")
        print('   • "Check my ETH balance"')
        print('   • "help" - for detailed instructions')
        print('   • "quit" - to exit')
        print(f"{'='*50}")

        while True:
            try:
                # Get user input
                command = input(f"\n SpoonOS EVM > ").strip()

                if not command:
                    continue

                # Process the command
                print(f"\n Processing: \"{command}\"...")

                result = await agent.process_command(
                    command=command,
                    network=agent.default_network,
                    debug=False
                )

                # Display result
                if result['success']:
                    print(f"✅ Success!")
                    if 'result' in result and isinstance(result['result'], dict):
                        if 'message' in result['result']:
                            print(f" {result['result']['message']}")

                    if result.get('transaction_hash'):
                        print(f" Transaction: {result['transaction_hash']}")
                        if result.get('explorer_link'):
                            print(f" View on explorer: {result['explorer_link']}")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")

            except KeyboardInterrupt:
                print("\n Session ended by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to start interactive session: {e}")

async def main():
    """Main entry point - run the demo by default."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await run_interactive_session()
    else:
        await run_interactive_demo()

if __name__ == "__main__":
    # Load environment variables if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    asyncio.run(main())