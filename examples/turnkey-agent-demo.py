"""
Turnkey Agent Demo - AI-powered secure blockchain operations

This demo shows how to use an AI agent to perform Turnkey operations:
- Sign transactions (single and batch)
- Message and EIP-712 signing
- Wallet and account management
- Activity monitoring and audit
- Complete transaction workflows

The agent automatically handles:
- Multi-account discovery
- Batch operations
- Gas optimization
- Error handling and troubleshooting

All test data is embedded for easy demonstration.

Usage:
    python examples/turnkey-agent-demo.py
"""

import asyncio
import os
import time
from typing import List
from dotenv import load_dotenv

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.chat import ChatBot
from pydantic import Field

from dotenv import load_dotenv
load_dotenv()

# Import Turnkey tools
from spoon_ai.tools.turnkey_tools import (
    SignEVMTransactionTool,
    SignMessageTool,
    SignTypedDataTool,
    BroadcastTransactionTool,
    BuildUnsignedEIP1559TxTool,
    CompleteTransactionWorkflowTool,
    ListWalletsTool,
    ListWalletAccountsTool,
    ListAllAccountsTool,
    GetWalletTool,
    CreateWalletTool,
    CreateWalletAccountsTool,
    BatchSignTransactionsTool,
    GetActivityTool,
    ListActivitiesTool,
    WhoAmITool,
)

load_dotenv()


class TurnkeyAgentDemo:
    """Turnkey Agent-based comprehensive demonstration"""

    # Embedded test data with all configurations
    TEST_DATA = {
        "network": "sepolia",
        "chain_id": 11155111,
        "rpc_url": "https://1rpc.io/sepolia",
        "explorer": "https://sepolia.etherscan.io",
        
        # Transaction templates
        "transaction_templates": {
            "simple_transfer": {
                "description": "Simple ETH transfer",
                "value_wei": "10000000000000000",  # 0.01 ETH
                "data_hex": "0x",
                "gas_limit": "21000"
            }
        },
        
        # EIP-712 templates
        "eip712_templates": {
            "simple_mail": {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                    ],
                    "Mail": [
                        {"name": "contents", "type": "string"}
                    ],
                },
                "primaryType": "Mail",
                "domain": {"name": "Turnkey", "version": "1", "chainId": 11155111},
                "message": {"contents": "Hello from Turnkey Agent Demo"},
            },
            "permit": {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"}
                    ],
                    "Permit": [
                        {"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"},
                        {"name": "value", "type": "uint256"},
                        {"name": "nonce", "type": "uint256"},
                        {"name": "deadline", "type": "uint256"}
                    ]
                },
                "primaryType": "Permit",
                "domain": {
                    "name": "Test Token",
                    "version": "1",
                    "chainId": 11155111,
                    "verifyingContract": "0x1234567890123456789012345678901234567890"
                },
                "message": {
                    "owner": "0x6B02D710B27bEB57156898d778079016404BEDC3",
                    "spender": "0x1234567890123456789012345678901234567890",
                    "value": "1000000000000000000",
                    "nonce": 0,
                    "deadline": 1735689600
                }
            }
        },
        
        # Message signing templates
        "message_templates": {
            "authentication": "Sign in to DApp at 2025-10-22",
            "order": "Sell 1.5 ETH at 2500 USDC - Order #12345",
            "simple": "Hello Turnkey Trading Agent"
        },
        
        # Batch operation settings
        "batch_settings": {
            "max_accounts": 3,
            "default_value_wei": "1000000000000000",  # 0.001 ETH
            "enable_broadcast": False
        },
        
        # Wallet configuration templates
        "wallet_templates": {
            "ethereum_wallet": {
                "accounts": [{
                    "curve": "CURVE_SECP256K1",
                    "pathFormat": "PATH_FORMAT_BIP32",
                    "path": "m/44'/60'/0'/0/0",
                    "addressFormat": "ADDRESS_FORMAT_ETHEREUM"
                }],
                "mnemonic_length": 24
            },
            "multi_account_wallet": {
                "accounts": [
                    {
                        "curve": "CURVE_SECP256K1",
                        "pathFormat": "PATH_FORMAT_BIP32",
                        "path": f"m/44'/60'/0'/0/{i}",
                        "addressFormat": "ADDRESS_FORMAT_ETHEREUM"
                    } for i in range(3)
                ],
                "mnemonic_length": 24
            }
        }
    }

    def __init__(self):
        """Initialize the demo with embedded test data"""
        self.load_test_data()
        self.agents = {}

    def load_test_data(self):
        """Load test data from embedded TEST_DATA configuration"""
        try:
            data = self.TEST_DATA
            
            # Load basic configuration
            self.network = data.get("network", "sepolia")
            self.chain_id = data.get("chain_id", 11155111)
            self.rpc_url = data.get("rpc_url", "")
            self.explorer = data.get("explorer", "")
            
            # Load transaction templates
            self.transaction_templates = data.get("transaction_templates", {})
            
            # Load EIP-712 templates
            self.eip712_templates = data.get("eip712_templates", {})
            
            # Load message templates
            self.message_templates = data.get("message_templates", {})
            
            # Load batch settings
            self.batch_settings = data.get("batch_settings", {})
            
            # Load wallet templates
            self.wallet_templates = data.get("wallet_templates", {})
            
            print(f"✅ Loaded test data from embedded configuration")
            print(f"   Network: {self.network} (Chain ID: {self.chain_id})")
            print(f"   RPC URL: {self.rpc_url}")
            print(f"   Explorer: {self.explorer}")
            print(f"   Transaction Templates: {len(self.transaction_templates)}")
            print(f"   EIP-712 Templates: {len(self.eip712_templates)}")
            print(f"   Message Templates: {len(self.message_templates)}")
            
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            # Set minimal defaults
            self.network = "sepolia"
            self.chain_id = 11155111
            self.rpc_url = "https://1rpc.io/sepolia"
            self.explorer = "https://sepolia.etherscan.io"
            self.transaction_templates = {}
            self.eip712_templates = {}
            self.message_templates = {}
            self.batch_settings = {}
            self.wallet_templates = {}

    def create_agent(self, name: str, tools: List, description: str) -> ToolCallAgent:
        """Create a specialized agent with specific tools"""
        
        network = self.network
        chain_id = self.chain_id
        rpc_url = self.rpc_url
        
        class TurnkeySpecializedAgent(ToolCallAgent):
            agent_name: str = name
            agent_description: str = description
            system_prompt: str = f"""
        You are a Turnkey secure blockchain specialist focused on {description}.

        **Environment Configuration:**
        - Network: {network} (Chain ID: {chain_id})
        - RPC URL: {rpc_url}

        **Key Operations and Best Practices:**

        1. **Transaction Workflow**:
        - Build unsigned tx → Sign with Turnkey → Optionally broadcast
        - Use complete_transaction_workflow for end-to-end operations
        - Always check account balance before broadcasting

        2. **Batch Operations**:
        - Use batch_sign_transactions for multi-account operations
        - Automatically discovers all organization accounts
        - Control with max_accounts and enable_broadcast parameters

        3. **Account Management**:
        - list_all_accounts: Discover all accounts across wallets
        - list_wallets: Get all wallets
        - create_wallet: Create new wallets dynamically

        4. **Signing Operations**:
        - sign_evm_transaction: Sign prepared transactions
        - sign_message: Sign arbitrary messages (authentication, orders)
        - sign_typed_data: Sign EIP-712 structured data (permits, approvals)

        5. **Security Best Practices**:
        - Never expose private keys
        - Verify transaction details before signing
        - Monitor activity logs for security
        - Test on testnet before mainnet

        **Tool Usage Examples:**

        - Complete workflow: complete_transaction_workflow(sign_with="0x...", to_address="0x...", value_wei="1000000", enable_broadcast=false)
        - Batch operations: batch_sign_transactions(to_address="0x...", value_wei="1000000", max_accounts="3", enable_broadcast=false)
        - Account discovery: list_all_accounts()

        Provide clear, informative responses based on the tool results.
        Always explain security implications and best practices.
        """
            max_steps: int = 20
            avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager(tools))
        
        agent = TurnkeySpecializedAgent(
            llm=ChatBot(
                llm_provider="openrouter",
                model_name="openai/gpt-4o"
            )
        )
        return agent

    def setup_agents(self):
        """Setup specialized agents for different Turnkey operations"""
        
        # Secure Signing Agent (4 tools)
        signing_tools = [
            ListAllAccountsTool(),  # Add this tool to get account addresses
            SignEVMTransactionTool(),
            SignMessageTool(),
            SignTypedDataTool(),
        ]
        self.agents['signing'] = self.create_agent(
            "Secure Signing Specialist",
            signing_tools,
            "Expert in secure transaction and message signing using Turnkey"
        )
        
        # Transaction Manager Agent (4 tools)
        transaction_tools = [
            BuildUnsignedEIP1559TxTool(),
            SignEVMTransactionTool(),
            BroadcastTransactionTool(),
            CompleteTransactionWorkflowTool(),
        ]
        self.agents['transaction'] = self.create_agent(
            "Transaction Manager",
            transaction_tools,
            "Specialist in building, signing, and broadcasting blockchain transactions"
        )
        
        # Account Manager Agent (6 tools)
        account_tools = [
            ListWalletsTool(),
            ListWalletAccountsTool(),
            ListAllAccountsTool(),
            GetWalletTool(),
            CreateWalletTool(),
            CreateWalletAccountsTool(),
        ]
        self.agents['account'] = self.create_agent(
            "Account Manager",
            account_tools,
            "Expert in managing Turnkey wallets and accounts"
        )
        
        # Batch Operations Agent (4 tools)
        batch_tools = [
            ListAllAccountsTool(),
            BatchSignTransactionsTool(),
            SignMessageTool(),
            SignTypedDataTool(),
        ]
        self.agents['batch'] = self.create_agent(
            "Batch Operations Manager",
            batch_tools,
            "Specialist in multi-account batch operations and parallel signing"
        )
        
        # Activity Monitor Agent (3 tools: GetActivityTool, ListActivitiesTool, WhoAmITool)
        monitor_tools = [
            GetActivityTool(),      # Get specific activity details by ID
            ListActivitiesTool(),   # List recent activities with pagination
            WhoAmITool(),           # Get organization information
        ]
        self.agents['monitor'] = self.create_agent(
            "Activity Monitor",
            monitor_tools,
            "Expert in monitoring Turnkey activities and organization status"
        )

    def print_section_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")

    async def demo_organization_info(self):
        """Demonstrate organization information capabilities"""
        self.print_section_header("1. Organization Information and Status")
        
        # Use persistent agent
        agent = self.agents['monitor']
        
        # Scenario 1: Get organization info
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Organization Status")
        print(f"{'-'*60}")
        
        response1 = await agent.run(
            "Get information about the Turnkey organization. What can you tell me about the current organization setup?"
        )
        print(f"✅ Response: {response1}")
        
        # Scenario 2: Recent activities with filtering
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Recent Activities with Filtering")
        print(f"{'-'*60}")
        
        response2 = await agent.run(
            "Show me the recent activities in the organization. Use the list_activities tool with limit=5 to get the latest 5 activities. What operations have been performed recently?"
        )
        print(f"✅ Response: {response2}")
        
        # Scenario 3: Filter by activity type
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Filter by Transaction Activities")
        print(f"{'-'*60}")
        
        response3 = await agent.run(
            "Filter activities to show only transaction signing activities. Use list_activities with filter_by_type=['ACTIVITY_TYPE_SIGN_TRANSACTION_V2'] and limit=3 to get only transaction signing activities."
        )
        print(f"✅ Response: {response3}")
        
        # Scenario 4: Filter by status
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Filter by Completed Status")
        print(f"{'-'*60}")
        
        response4 = await agent.run(
            "Show me only completed activities. Use list_activities with filter_by_status=['ACTIVITY_STATUS_COMPLETED'] and limit=3 to get only successfully completed activities."
        )
        print(f"✅ Response: {response4}")

        # Scenario 5: Filter by failed status
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Filter by Failed Status")
        print(f"{'-'*60}")
        
        response5 = await agent.run(
            "Show me only failed activities. Use list_activities with filter_by_status=['ACTIVITY_STATUS_FAILED'] and limit=3 to get only failed activities."
        )
        print(f"✅ Response: {response5}")
        
        # Scenario 5: Get specific activity details
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Activity Details")
        print(f"{'-'*60}")
        
        response5 = await agent.run(
            "First, get the list of recent activities. Then, from that list, extract the activity ID (the ID field) of the first (most recent) activity and use that exact ID to get detailed information about that specific activity."
        )
        print(f"✅ Response: {response5}")

    async def demo_account_management(self):
        """Demonstrate account management capabilities"""
        self.print_section_header("2. Account and Wallet Management")
        
        # Use persistent agent
        agent = self.agents['account']
        
        # Scenario 1: List all wallets
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Wallet Discovery")
        print(f"{'-'*60}")
        
        response1 = await agent.run(
            "List all wallets in the organization. What wallets are available and what are their names?"
        )
        print(f"✅ Response: {response1}")
        
        # Scenario 2: List all accounts with limit
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Account Discovery with Limit")
        print(f"{'-'*60}")
        
        response2 = await agent.run(
            "List 10 accounts in all wallets. Show me the full list of accounts, including wallet name, address, and fork path."
        )
        print(f"✅ Response: {response2}")
        
        # Scenario 3: Get specific wallet details
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Wallet Details")
        print(f"{'-'*60}")
        
        response3 = await agent.run(
            "First, list all wallets to get a wallet ID. Then use get_wallet tool with that wallet ID to get detailed information about the first wallet."
        )
        print(f"✅ Response: {response3}")
        
        # Scenario 4: List accounts for specific wallet
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Wallet Account Listing")
        print(f"{'-'*60}")
        
        response4 = await agent.run(
            "First, list all wallets to get a wallet ID. Then use list_wallet_accounts tool with that wallet ID and limit=5 to get accounts for a specific wallet."
        )
        print(f"✅ Response: {response4}")

        # Scenario 5: Add accounts to existing wallet (temporarily disabled due to API issues)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Add Accounts to Existing Wallet")
        print(f"{'-'*60}")
        
        # A unique path is generated using the timestamp
        unique_index = int(time.time()) % 1000  # Use the last 3 bits of the timestamp as the index
        response5 = await agent.run(
            f"First, list all wallets to get a wallet ID. Then use create_wallet_accounts tool to add a new Ethereum account to that wallet. Use accounts_json='[{{\"curve\":\"CURVE_SECP256K1\",\"pathFormat\":\"PATH_FORMAT_BIP32\",\"path\":\"m/44\\'/60\\'/0\\'/0/{unique_index}\",\"addressFormat\":\"ADDRESS_FORMAT_ETHEREUM\"}}]'"
        )
        print(f"✅ Response: {response5}")
        
        # Scenario 6: Create new wallet with unique name
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Create New Wallet")
        print(f"{'-'*60}")
        
        unique_name = f"Demo_Wallet_{int(time.time())}"
        response6 = await agent.run(
            f"Create a new wallet called '{unique_name}' with a single Ethereum account. Use create_wallet with wallet_name='{unique_name}' and mnemonic_length='24'."
        )
        print(f"✅ Response: {response6}")
        
    async def demo_secure_signing(self):
        """Demonstrate secure signing capabilities"""
        self.print_section_header("3. Secure Signing Operations")
        
        # Use persistent agent
        agent = self.agents['signing']
        
        # Scenario 1: Message signing
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Message Signing")
        print(f"{'-'*60}")
        
        # Get signing identity from environment variable
        sign_with = os.getenv("TURNKEY_SIGN_WITH")
        
        if not sign_with:
            response1 = await agent.run(f"""You need to sign the message 'Hello Turnkey'. Follow these exact steps:

            Step 1: Call list_all_accounts tool to get available accounts
            Step 2: Extract the first account address from the response
            Step 3: Use that account address as sign_with parameter in sign_message tool
            Step 4: Call sign_message tool with:
               - sign_with: [the account address from step 2]
               - message: 'Hello Turnkey'
               - use_keccak256: true
            
            Do NOT use placeholder addresses. You must get a real account address first.""")
        else:
            response1 = await agent.run(f"""Sign the message 'Hello Turnkey' using the configured signing identity.

            Use sign_message tool with:
            - sign_with: '{sign_with}'
            - message: 'Hello Turnkey'
            - use_keccak256: true
            
            Explain what message signing is used for and the security benefits.""")
        print(f"✅ Response: {response1}")
        
        # Scenario 2: Message signing with Keccak256
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Message Signing with Keccak256")
        print(f"{'-'*60}")
        
        if not sign_with:
            response2 = await agent.run(f"""Sign the message 'Trading Order #12345' using the same account from the previous step.

            Use sign_message tool with:
            - sign_with: [use the same account address from previous step]
            - message: 'Trading Order #12345'
            - use_keccak256: true""")
        else:
            response2 = await agent.run(f"""Sign the message 'Trading Order #12345' using the configured signing identity.

            Use sign_message tool with:
            - sign_with: '{sign_with}'
            - message: 'Trading Order #12345'
            - use_keccak256: true""")
        print(f"✅ Response: {response2}")
        
        # Scenario 3: EIP-712 simple mail signing
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: EIP-712 Simple Mail Signing")
        print(f"{'-'*60}")
        
        eip712_data = self.eip712_templates.get("simple_mail", {})
        if not sign_with:
            response3 = await agent.run(f"""Sign the following EIP-712 structured data using the same account from previous steps: {eip712_data}

            Use sign_typed_data tool with:
            - sign_with: [use the same account address from previous steps]
            - typed_data: {eip712_data}
            
            Explain what EIP-712 signing is and why it's important""")
        else:
            response3 = await agent.run(f"""Sign the following EIP-712 structured data using the configured signing identity: {eip712_data}

            Use sign_typed_data tool with:
            - sign_with: '{sign_with}'
            - typed_data: {eip712_data}
            
            Explain what EIP-712 signing is and why it's important""")
        print(f"✅ Response: {response3}")
        
        # Scenario 4: EIP-712 permit signing
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: EIP-712 Permit Signing")
        print(f"{'-'*60}")
        
        permit_data = self.eip712_templates.get("permit", {})
        if not sign_with:
            response4 = await agent.run(f"""Sign the following EIP-712 permit data using the same account: {permit_data}

            Use sign_typed_data tool with:
            - sign_with: [use the same account address from previous steps]
            - typed_data: {permit_data}
            
            Explain how permits work for gasless token approvals""")
        else:
            response4 = await agent.run(f"""Sign the following EIP-712 permit data using the configured signing identity: {permit_data}

            Use sign_typed_data tool with:
            - sign_with: '{sign_with}'
            - typed_data: {permit_data}
            
            Explain how permits work for gasless token approvals""")
        print(f"✅ Response: {response4}")

    async def demo_transaction_operations(self):
        """Demonstrate transaction building and execution"""
        self.print_section_header("4. Transaction Operations")
        
        # Use persistent agent
        agent = self.agents['transaction']
        
        # Scenario 1: Complete transaction workflow
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Complete Transaction Workflow")
        print(f"{'-'*60}")
        
        # # Get signing identity from environment variable
        sign_with = os.getenv("TURNKEY_SIGN_WITH")
        
        if not sign_with:
            response1 = await agent.run(f"""Perform a complete transaction workflow on {self.network}:

            Steps:
            1. Build unsigned transaction:
            - Use the first available account as sender
            - Send to: 0xF3D4D6320dd41f14B8Fa6550a6F33c46c6F41111
            - Value: 1000000000000000 wei (0.001 ETH)
            - Use complete_transaction_workflow tool

            2. Explain each step of the process

            Show me the transaction details and signing result.""")
        else:
            response1 = await agent.run(f"""Perform a complete transaction workflow on {self.network}:

            Steps:
            1. Build unsigned transaction:
            - Use configured signing identity: {sign_with}
            - Send to: 0xF3D4D6320dd41f14B8Fa6550a6F33c46c6F41111
            - Value: 10000000000000000 wei (0.01 ETH)
            - Use complete_transaction_workflow tool
            - enable_broadcast: true

            2. Explain each step of the process

        # Show me the transaction details and signing result.""")
        print(f"✅ Response: {response1}")
        
        # Scenario 2: Build unsigned transaction with custom gas
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Build Transaction with Custom Gas")
        print(f"{'-'*60}")
        
        if not sign_with:
            response2 = await agent.run(f"""Build an unsigned EIP-1559 transaction with custom gas settings:

            - from_addr: Use the first available account
            - to_addr: 0xF3D4D6320dd41f14B8Fa6550a6F33c46c6F41111
            - value_wei: 5000000000000000 (0.005 ETH)
            - priority_gwei: 2
            - max_fee_gwei: 20
            - gas_limit: 25000

            Explain the gas fee structure and how it affects transaction costs.""")
        else:
            response2 = await agent.run(f"""Build an unsigned EIP-1559 transaction with custom gas settings:

            - from_addr: {sign_with}
            - to_addr: 0xF3D4D6320dd41f14B8Fa6550a6F33c46c6F41111
            - value_wei: 5000000000000000 (0.005 ETH)
            - priority_gwei: 2
            - max_fee_gwei: 20
            - gas_limit: 25000

            Explain the gas fee structure and how it affects transaction costs.""")
        print(f"✅ Response: {response2}")
        
        # Scenario 3: Sign and broadcast separately
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Separate Sign and Broadcast")
        print(f"{'-'*60}")
        
        if not sign_with:
            response3 = await agent.run(f"""Demonstrate separate signing and broadcasting:

            1. First, build an unsigned transaction using build_unsigned_eip1559_tx with from_addr: Use the first available account
            2. Then sign it using sign_evm_transaction
            3. Finally, broadcast it using broadcast_transaction 
            This shows the modular approach to transaction handling.""")
        else:
            response3 = await agent.run(f"""Demonstrate separate signing and broadcasting:

            1. First, build an unsigned transaction using build_unsigned_eip1559_tx with from_addr: {sign_with}
            2. Then sign it using sign_evm_transaction
            3. Finally, broadcast it using broadcast_transaction 

            This shows the modular approach to transaction handling.""")
        print(f"✅ Response: {response3}")
        
    async def demo_batch_operations(self):
        """Demonstrate batch operations for multiple accounts"""
        self.print_section_header("5. Batch Operations - Multi-Account Signing")
        
        # Use persistent agent
        agent = self.agents['batch']
        
        # Scenario 1: Discover all accounts first
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Account Discovery")
        print(f"{'-'*60}")
        
        response1 = await agent.run(
            "List all accounts in the organization with limit=10. How many accounts are available for batch operations?"
        )
        print(f"✅ Response: {response1}")
        
        # Scenario 2: Batch sign transactions with custom parameters
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Batch Transaction Signing")
        print(f"{'-'*60}")
        
        response2 = await agent.run(f"""Perform batch transaction signing for all accounts:

        Call batch_sign_transactions with parameters:
        - to_address: "0xF3D4D6320dd41f14B8Fa6550a6F33c46c6F41111"
        - value_wei: "1000000000000000"  (0.001 ETH per account)
        - max_accounts: "2"
        - enable_broadcast: true

        Explain:
        1. How many accounts were processed
        2. The signing results for each account
        3. Why batch operations are useful for trading agents""")
        print(f"✅ Response: {response2}")
        
        
        # Scenario 3: Batch message signing
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Batch Message Signing")
        print(f"{'-'*60}")
        
        response3 = await agent.run(f"""Sign the same message with multiple accounts:

        Use sign_message tool to sign 'Portfolio Rebalancing Order #789' with different accounts.
        This demonstrates how batch message signing works for trading operations.

        Explain the use cases for batch message signing in trading systems.""")
        print(f"✅ Response: {response3}")
        
        # Scenario 4: Batch EIP-712 signing
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Batch EIP-712 Signing")
        print(f"{'-'*60}")
        
        # Get EIP-712 permit data from templates
        permit_data = self.eip712_templates.get("permit", {})
        
        response4 = await agent.run(f"""Sign EIP-712 structured data with multiple accounts:

        Use sign_typed_data tool to sign the following permit data with different accounts:
        {permit_data}
        
        This shows how batch EIP-712 signing works for token permits and approvals.

        Explain the benefits of batch EIP-712 signing for DeFi operations.""")
        print(f"✅ Response: {response4}")

    async def run_comprehensive_demo(self):
        """Run the complete agent-based demonstration"""
        print("🔐 Turnkey SDK - AI Agent Comprehensive Demonstration")
        print("=" * 80)
        print("This demo showcases Turnkey SDK tools through specialized AI agents")
        print("Each agent is an expert in specific aspects of secure blockchain operations")
        print("=" * 80)
        print(f" Network: {self.network}")
        print(f" Chain ID: {self.chain_id}")
        print(f" RPC URL: {self.rpc_url}")
        print(f" Test Data: {len(self.transaction_templates)} tx templates, {len(self.eip712_templates)} EIP-712 templates")

        try:
            # Setup all specialized agents
            print("\n🔧 Setting up specialized agents...")
            self.setup_agents()
            print(f"✅ Created {len(self.agents)} specialized agents")

            # Run comprehensive demonstrations
            await self.demo_organization_info()
            await self.demo_account_management()
            await self.demo_secure_signing()
            await self.demo_transaction_operations()
            await self.demo_batch_operations()

            # Final summary
            self.print_section_header("Demo Completed Successfully")
            for agent_name, agent in self.agents.items():
                tool_count = len(agent.avaliable_tools.tools)
                print(f"  ✅ {agent.agent_name}: {tool_count} specialized tools")

            total_tools = sum(len(agent.avaliable_tools.tools) for agent in self.agents.values())
            print(f"\n🔧 Total Tools Demonstrated: 16 Turnkey tools")
            print("   All demonstrations powered by AI agents with domain expertise")
            print("   Each agent provides intelligent analysis and secure operations")
            print("\nAgent Capabilities:")
            print("   1. Secure Signing Specialist: Message and EIP-712 signing")
            print("   2. Transaction Manager: Complete transaction workflows")
            print("   3. Account Manager: Wallet and account management")
            print("   4. Batch Operations Manager: Multi-account operations")
            print("   5. Activity Monitor: Security monitoring and audit")
            print("\nKey Features Demonstrated:")
            print("   ✅ Secure signing without private key exposure")
            print("   ✅ Multi-account batch operations")
            print("   ✅ Complete transaction workflows (build, sign, broadcast)")
            print("   ✅ EIP-712 structured data signing")
            print("   ✅ Activity monitoring and audit trails")
            print("   ✅ Dynamic wallet and account management")

        except Exception as e:
            print(f"\n❌ Demo error: {str(e)}")
            print("Please check your environment setup and Turnkey configuration")


async def main():
    """Main demonstration function"""
    print("\n🔐 Turnkey SDK - AI Agent Demonstration")
    print("=" * 80)
    print("Showcasing comprehensive Turnkey operations through specialized AI agents")
    print("Each agent is equipped with domain-specific tools and security expertise")
    print("=" * 80)

    demo = TurnkeyAgentDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())

