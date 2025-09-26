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
from spoon_ai.agents import EvmAgent
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.chat import ChatBot

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

        # Demo/Test addresses from environment variables
        self.demo_addresses = {
            "primary_wallet": os.getenv("PRIMARY_WALLET", None),
            "secondary_wallet": os.getenv("SECONDARY_WALLET", None),
        }

        # Demo private key from environment variable
        self.private_key = os.getenv("PRIVATE_KEY")

        # Available tools - use the EVM agent's available tools
        self.available_tools = self.evm_agent.avaliable_tools

    async def connect(self):
        """Connect method required by SpoonReactMCP base class."""
        return True

    async def initialize(self):
        """Initialize the agent and EVM tools."""
        await super().initialize()
        await self.evm_agent.initialize()

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
        private_key = private_key or os.getenv("PRIVATE_KEY")

        try:
            # Execute the command using EVM agent
            result = await self.evm_agent.process_evm_command(
                command=command,
                network=network,
                private_key=private_key,
                confirm=confirm,
                **kwargs
            )

            # Process the result
            execution_time = (datetime.now() - start_time).total_seconds()

            if not result.get("success", False):
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "command": command,
                    "network": network_config.name,
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time.isoformat()
                }
            else:
                # Extract transaction hash if present
                tx_hash = None
                explorer_link = None
                result_data = result.get("result", "")

                # Try to extract transaction hash from result
                if isinstance(result_data, str):
                    import re
                    # Look for transaction hash patterns in the response
                    hash_pattern = r"0x[a-fA-F0-9]{64}"
                    tx_match = re.search(hash_pattern, result_data)
                    if tx_match:
                        tx_hash = tx_match.group(0)

                # Generate explorer link if we have a transaction hash
                if tx_hash:
                    explorer_link = f"{network_config.explorer_url}/tx/{tx_hash}"

                return {
                    "success": True,
                    "result": {"message": result_data},
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


async def run_interactive_demo():
    """Run an interactive demonstration of the EVM Command Agent."""

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

        for i, demo in enumerate(demo_commands, 1):
            print(f"\n[{i}/{len(demo_commands)}] {demo['description']}")
            print(f"Command: \"{demo['command']}\" (Network: {demo['network']})")
            print("-" * 50)

            try:
                result = await agent.process_command(
                    command=demo['command'],
                    network=demo['network'],
                    private_key=agent.private_key,  # Use demo private key
                    confirm=False,  # Skip confirmations in demo
                    debug=False  # Disable debug to reduce log noise
                )

                if result['success']:
                    print("✅ SUCCESS")

                    # Print detailed result information
                    result_data = result.get('result', {})
                    if isinstance(result_data, dict) and 'message' in result_data:
                        message = result_data['message']
                        print(f"Result: {message}")
                    elif result_data:
                        print(f"Result: {result_data}")

                    # Print transaction details if available
                    if result.get('transaction_hash'):
                        print(f"Transaction: {result['transaction_hash']}")

                    # Print explorer link if available
                    if result.get('explorer_link'):
                        print(f"Explorer: {result['explorer_link']}")

                    # Print execution time
                    exec_time = result.get('execution_time_seconds')
                    if exec_time:
                        print(f"Execution time: {exec_time:.2f}s")

                    # Print network info
                    if result.get('network'):
                        print(f"Network: {result['network']}")

                else:
                    print("❌ FAILED")
                    error_msg = str(result.get('error', 'Unknown error'))
                    print(f"Error: {error_msg[:200]}{'...' if len(error_msg) > 200 else ''}")

                    # Print additional error context
                    if result.get('network'):
                        print(f"Network: {result['network']}")
                    if result.get('execution_time_seconds'):
                        print(f"Failed after: {result['execution_time_seconds']:.2f}s")

            except Exception as e:
                print(f"❌ Exception: {str(e)[:100]}")

            # Small delay for readability
            await asyncio.sleep(2)


    except KeyboardInterrupt:
        print("\nDemo interrupted")
    except Exception as e:
        print(f"\nDemo failed: {e}")

async def main():
    """Main entry point - run the demo."""
    await run_interactive_demo()

if __name__ == "__main__":
    # Load environment variables if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    asyncio.run(main())