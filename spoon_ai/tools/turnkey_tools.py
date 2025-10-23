"""
Turnkey Tools - Secure Blockchain Operations

This module provides Turnkey SDK tools for secure blockchain operations including:
- Transaction signing and broadcasting
- Message and EIP-712 signing
- Multi-account management
- Activity audit and monitoring
- Wallet and account operations
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from spoon_ai.tools.base import BaseTool
from spoon_ai.turnkey import Turnkey


class TurnkeyBaseTool(BaseTool):
    """Base class for Turnkey tools with shared client initialization"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Turnkey client"""
        if self._client is None:
            try:
                self._client = Turnkey()
            except Exception as e:
                raise ValueError(f"Failed to initialize Turnkey client: {e}")
        return self._client


class SignEVMTransactionTool(TurnkeyBaseTool):
    """Sign EVM transaction using Turnkey"""
    
    name: str = "sign_evm_transaction"
    description: str = "Sign EVM transaction using Turnkey secure signing"
    parameters: dict = {
        "type": "object",
        "properties": {
            "sign_with": {
                "type": "string",
                "description": "Signing identity (wallet account address / private key address / private key ID)"
            },
            "unsigned_tx": {
                "type": "string", 
                "description": "Raw unsigned transaction (hex string starting with 0x02 for EIP-1559)"
            }
        },
        "required": ["sign_with", "unsigned_tx"]
    }
    
    async def execute(self, sign_with: str, unsigned_tx: str, **kwargs) -> str:
        """Sign EVM transaction"""
        try:
            result = self.client.sign_evm_transaction(
                sign_with=sign_with,
                unsigned_tx=unsigned_tx
            )
            
            signed_tx = (
                result.get("activity", {})
                .get("result", {})
                .get("signTransactionResult", {})
                .get("signedTransaction")
            )
            
            if signed_tx:
                return f"✅ Transaction signed successfully\nSigned Transaction: {signed_tx}"
            else:
                return f"❌ Signing failed. Response: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            return f"❌ Signing failed: {str(e)}"


class SignMessageTool(TurnkeyBaseTool):
    """Sign arbitrary message using Turnkey"""
    
    name: str = "sign_message"
    description: str = "Sign arbitrary message using Turnkey secure signing"
    parameters: dict = {
        "type": "object",
        "properties": {
            "sign_with": {
                "type": "string",
                "description": "Signing identity (wallet account address / private key address / private key ID)"
            },
            "message": {
                "type": "string",
                "description": "Message to sign (text or bytes, bytes will be decoded as UTF-8)"
            },
            "use_keccak256": {
                "type": "boolean",
                "description": "Whether to use KECCAK256 hash (default: true, follows Ethereum convention)",
                "default": True
            }
        },
        "required": ["sign_with", "message"]
    }
    
    async def execute(self, sign_with: str, message: str, use_keccak256: bool = True, **kwargs) -> str:
        """Sign message"""
        try:
            result = self.client.sign_message(
                sign_with=sign_with,
                message=message,
                use_keccak256=use_keccak256
            )
            
            status = result.get("activity", {}).get("status")
            if status == "ACTIVITY_STATUS_COMPLETED":
                return f"✅ Message signed successfully\nStatus: {status}"
            else:
                return f"❌ Signing failed. Status: {status}\nResponse: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            return f"❌ Message signing failed: {str(e)}"


class SignTypedDataTool(TurnkeyBaseTool):
    """Sign EIP-712 structured data using Turnkey"""
    
    name: str = "sign_typed_data"
    description: str = "Sign EIP-712 structured data using Turnkey secure signing"
    parameters: dict = {
        "type": "object",
        "properties": {
            "sign_with": {
                "type": "string",
                "description": "Signing identity (wallet account address / private key address / private key ID)"
            },
            "typed_data": {
                "type": "object",
                "description": "EIP-712 structured data (domain/types/message) or JSON string"
            }
        },
        "required": ["sign_with", "typed_data"]
    }
    
    async def execute(self, sign_with: str, typed_data: dict, **kwargs) -> str:
        """Sign EIP-712 typed data"""
        try:
            result = self.client.sign_typed_data(
                sign_with=sign_with,
                typed_data=typed_data
            )
            
            status = result.get("activity", {}).get("status")
            if status == "ACTIVITY_STATUS_COMPLETED":
                return f"✅ EIP-712 data signed successfully\nStatus: {status}"
            else:
                return f"❌ EIP-712 signing failed. Status: {status}\nResponse: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            return f"❌ EIP-712 signing failed: {str(e)}"


class BroadcastTransactionTool(TurnkeyBaseTool):
    """Broadcast signed transaction to blockchain"""
    
    name: str = "broadcast_transaction"
    description: str = "Broadcast signed transaction to blockchain network"
    parameters: dict = {
        "type": "object",
        "properties": {
            "signed_tx": {
                "type": "string",
                "description": "Signed transaction hex string"
            },
            "rpc_url": {
                "type": "string",
                "description": "RPC URL for blockchain network (optional, uses WEB3_RPC_URL from env)"
            }
        },
        "required": ["signed_tx"]
    }
    
    async def execute(self, signed_tx: str, rpc_url: str = None, **kwargs) -> str:
        """Broadcast transaction"""
        try:
            from web3 import Web3
            from eth_utils import to_bytes
            
            # Use provided RPC or environment variable
            rpc = rpc_url or os.getenv("WEB3_RPC_URL")
            if not rpc:
                return "❌ RPC URL not provided. Set WEB3_RPC_URL environment variable or provide rpc_url parameter."
            
            w3 = Web3(Web3.HTTPProvider(rpc))
            
            # Inject PoA middleware for compatibility
            try:
                from web3.middleware import ExtraDataToPOAMiddleware
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except ImportError:
                from web3.middleware import geth_poa_middleware
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not w3.is_connected():
                return "❌ Cannot connect to RPC endpoint"
            
            # Broadcast transaction
            tx_hash = w3.eth.send_raw_transaction(to_bytes(hexstr=signed_tx))
            tx_hash_hex = tx_hash.hex()
            
            # Determine explorer URL
            chain_id = w3.eth.chain_id
            if chain_id == 1:
                explorer_url = f"https://etherscan.io/tx/{tx_hash_hex}"
            elif chain_id == 11155111:  # Sepolia
                explorer_url = f"https://sepolia.etherscan.io/tx/{tx_hash_hex}"
            elif chain_id == 5:  # Goerli
                explorer_url = f"https://goerli.etherscan.io/tx/{tx_hash_hex}"
            else:
                explorer_url = f"Chain {chain_id} - TxHash: {tx_hash_hex}"
            
            return f"✅ Transaction broadcasted successfully\nTxHash: {tx_hash_hex}\nExplorer: {explorer_url}"
            
        except ImportError:
            return "❌ web3 and eth-utils not installed. Install with: pip install web3 eth-utils"
        except Exception as e:
            return f"❌ Broadcast failed: {str(e)}"


class ListWalletsTool(TurnkeyBaseTool):
    """List all wallets in the organization"""
    
    name: str = "list_wallets"
    description: str = "List all wallets in the Turnkey organization"
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    async def execute(self, **kwargs) -> str:
        """List wallets"""
        try:
            result = self.client.list_wallets()
            wallets = result.get("wallets", [])
            
            if not wallets:
                return "ℹ️ No wallets found in the organization"
            
            wallet_list = []
            for wallet in wallets:
                wallet_list.append(f"• {wallet.get('walletName', 'Unnamed')} (ID: {wallet.get('walletId', 'Unknown')})")
            
            return f"✅ Found {len(wallets)} wallets:\n" + "\n".join(wallet_list)
            
        except Exception as e:
            return f"❌ Failed to list wallets: {str(e)}"


class ListWalletAccountsTool(TurnkeyBaseTool):
    """List accounts for a specific wallet"""
    
    name: str = "list_wallet_accounts"
    description: str = "List all accounts for a specific wallet"
    parameters: dict = {
        "type": "object",
        "properties": {
            "wallet_id": {
                "type": "string",
                "description": "Wallet ID"
            },
            "limit": {
                "type": "string",
                "description": "Number of accounts to return (optional, no limit if not specified)"
            },
            "before": {
                "type": "string",
                "description": "Pagination cursor, returns accounts before this ID (optional)"
            },
            "after": {
                "type": "string",
                "description": "Pagination cursor, returns accounts after this ID (optional)"
            }
        },
        "required": ["wallet_id"]
    }
    
    async def execute(self, wallet_id: str, limit: str = None, before: str = None, after: str = None, **kwargs) -> str:
        """List wallet accounts"""
        try:
            result = self.client.list_wallet_accounts(wallet_id, limit=limit, before=before, after=after)
            accounts = result.get("accounts", [])
            
            if not accounts:
                return f"ℹ️ No accounts found for wallet {wallet_id}"
            
            account_list = []
            for account in accounts:
                address = account.get("address", "Unknown")
                path = account.get("path", "Unknown")
                account_list.append(f"• {address} (Path: {path})")
            
            return f"✅ Found {len(accounts)} accounts for wallet {wallet_id}:\n" + "\n".join(account_list)
            
        except Exception as e:
            return f"❌ Failed to list wallet accounts: {str(e)}"


class GetActivityTool(TurnkeyBaseTool):
    """Get activity details by ID"""
    
    name: str = "get_activity"
    description: str = "Get detailed information about a specific activity"
    parameters: dict = {
        "type": "object",
        "properties": {
            "activity_id": {
                "type": "string",
                "description": "Activity ID"
            }
        },
        "required": ["activity_id"]
    }
    
    async def execute(self, activity_id: str, **kwargs) -> str:
        """Get activity details"""
        try:
            result = self.client.get_activity(activity_id)
            
            activity = result.get("activity", {})
            activity_type = activity.get("type", "Unknown")
            status = activity.get("status", "Unknown")
            created_at = activity.get("createdAt", "Unknown")
            
            return f"✅ Activity Details:\nType: {activity_type}\nStatus: {status}\nCreated: {created_at}\n\nFull Response:\n{json.dumps(result, indent=2)}"
            
        except Exception as e:
            return f"❌ Failed to get activity: {str(e)}"


class ListActivitiesTool(TurnkeyBaseTool):
    """List recent activities in the organization"""
    
    name: str = "list_activities"
    description: str = "List recent activities in the Turnkey organization"
    parameters: dict = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "string",
                "description": "Number of activities to return (optional, default: 10)"
            },
            "before": {
                "type": "string",
                "description": "Pagination cursor, returns activities before this ID (optional)"
            },
            "after": {
                "type": "string",
                "description": "Pagination cursor, returns activities after this ID (optional)"
            },
            "filter_by_status": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by activity status (optional). Examples: ['ACTIVITY_STATUS_COMPLETED', 'ACTIVITY_STATUS_FAILED']"
            },
            "filter_by_type": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by activity type (optional). Examples: ['ACTIVITY_TYPE_SIGN_TRANSACTION_V2', 'ACTIVITY_TYPE_CREATE_ACCOUNT']"
            }
        },
        "required": []
    }
    
    async def execute(self, limit: str = "10", before: str = None, after: str = None, 
                     filter_by_status: list = None, filter_by_type: list = None, **kwargs) -> str:
        """List activities"""
        try:
            result = self.client.list_activities(
                limit=limit, 
                before=before, 
                after=after,
                filter_by_status=filter_by_status,
                filter_by_type=filter_by_type
            )
            activities = result.get("activities", [])
            
            if not activities:
                return "ℹ️ No recent activities found"
            
            activity_list = []
            for i, activity in enumerate(activities, 1):
                # Extract basic info
                activity_id = activity.get("id", "Unknown")
                activity_type = activity.get("type", "Unknown")
                status = activity.get("status", "Unknown")
                created_at = activity.get("createdAt", {})
                
                # Format timestamp
                if isinstance(created_at, dict) and "seconds" in created_at:
                    import datetime
                    timestamp = datetime.datetime.fromtimestamp(int(created_at["seconds"]))
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = str(created_at)
                
                # Extract key details based on activity type
                details = []
                intent = activity.get("intent", {})
                
                if "signTransactionIntentV2" in intent:
                    sign_with = intent["signTransactionIntentV2"].get("signWith", "Unknown")
                    details.append(f"SignWith: {sign_with}")
                elif "signMessageIntent" in intent:
                    message = intent["signMessageIntent"].get("message", "Unknown")
                    details.append(f"Message: {message[:30]}...")
                elif "createAccountIntent" in intent:
                    account_name = intent["createAccountIntent"].get("accountName", "Unknown")
                    details.append(f"Account: {account_name}")
                
                # Extract user info
                votes = activity.get("votes", [])
                if votes:
                    user = votes[0].get("user", {})
                    user_name = user.get("userName", "Unknown")
                    details.append(f"User: {user_name}")
                
                # Format the line
                details_str = " | ".join(details) if details else ""
                activity_list.append(f"{i}. {activity_type} | {status} | {time_str} | {details_str}")
            
            return f"✅ Recent Activities ({len(activities)}):\n" + "\n".join(activity_list)
            
        except Exception as e:
            return f"❌ Failed to list activities: {str(e)}"


class WhoAmITool(TurnkeyBaseTool):
    """Get organization information"""
    
    name: str = "whoami"
    description: str = "Get Turnkey organization information"
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    async def execute(self, **kwargs) -> str:
        """Get organization info"""
        try:
            result = self.client.whoami()
            
            org_info = result.get("organization", {})
            org_name = org_info.get("name", "Unknown")
            org_id = org_info.get("organizationId", "Unknown")
            
            return f"✅ Organization Information:\nName: {org_name}\nID: {org_id}\n\nFull Response:\n{json.dumps(result, indent=2)}"
            
        except Exception as e:
            return f"❌ Failed to get organization info: {str(e)}"


class BuildUnsignedEIP1559TxTool(BaseTool):
    """Build unsigned EIP-1559 transaction (supports NeoX)"""
    
    name: str = "build_unsigned_eip1559_tx"
    description: str = "Build unsigned EIP-1559 transaction for signing. Automatically detects and supports NeoX network (uses Policy Contract for gas fees)"
    parameters: dict = {
        "type": "object",
        "properties": {
            "from_addr": {
                "type": "string",
                "description": "From address (required)"
            },
            "to_addr": {
                "type": "string",
                "description": "To address (optional, defaults to from_addr for self-transfer)"
            },
            "value_wei": {
                "type": "string",
                "description": "Value in wei (default: 0)"
            },
            "data_hex": {
                "type": "string",
                "description": "Transaction data hex (default: 0x)"
            },
            "priority_gwei": {
                "type": "string",
                "description": "Priority fee in gwei (default: 1, ignored for NeoX networks)"
            },
            "max_fee_gwei": {
                "type": "string",
                "description": "Max fee per gas in gwei (optional, auto-calculated if not provided, ignored for NeoX)"
            },
            "gas_limit": {
                "type": "string",
                "description": "Gas limit (optional, auto-estimated if not provided)"
            },
            "rpc_url": {
                "type": "string",
                "description": "RPC URL (optional, uses WEB3_RPC_URL from environment)"
            }
        },
        "required": ["from_addr"]
    }
    
    def _get_neox_networks(self):
        """Get NeoX network configuration"""
        return {
            12227332: {
                "name": "NeoX Testnet",
                "policy_contract": "0x49cf4f5033b55b4E1700bC0D4764B5D6F42D3E3C",
                "gas_token": "0x1CE16390FD09040486221e912B87551E4e44Ab17",
            }
        }
    
    def _get_policy_contract_abi(self):
        """Get Policy Contract ABI"""
        return [
            {
                "constant": True,
                "inputs": [],
                "name": "getBaseFeePerGas",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "getMinPriorityFeePerGas",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
    
    async def execute(self, from_addr: str, to_addr: str = None, value_wei: str = "0", 
                     data_hex: str = "0x", priority_gwei: str = "1", max_fee_gwei: str = None,
                     gas_limit: str = None, rpc_url: str = None, **kwargs) -> str:
        """Build unsigned transaction (auto-detects NeoX)"""
        try:
            from web3 import Web3
            import rlp
            
            # Use provided RPC or environment variable
            rpc = rpc_url or os.getenv("WEB3_RPC_URL")
            if not rpc:
                return "❌ RPC URL not provided. Set WEB3_RPC_URL environment variable or provide rpc_url parameter."
            
            w3 = Web3(Web3.HTTPProvider(rpc))
            
            # Inject PoA middleware for compatibility
            try:
                from web3.middleware import ExtraDataToPOAMiddleware
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except ImportError:
                from web3.middleware import geth_poa_middleware
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not w3.is_connected():
                return "❌ Cannot connect to RPC endpoint"
            
            # Convert parameters
            to_addr = to_addr or from_addr
            value_wei = int(value_wei)
            priority_gwei = int(priority_gwei)
            
            # Get chain info
            chain_id = w3.eth.chain_id
            nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(from_addr))
            
            # Check if this is NeoX network
            neox_networks = self._get_neox_networks()
            is_neox = chain_id in neox_networks
            network_info = ""
            
            if is_neox:
                # NeoX: Get fees from Policy Contract
                neox_config = neox_networks[chain_id]
                network_info = f"🔷 {neox_config['name']} detected\n"
                
                try:
                    policy_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(neox_config['policy_contract']),
                        abi=self._get_policy_contract_abi()
                    )
                    base_fee = policy_contract.functions.getBaseFeePerGas().call()
                    min_priority_fee = policy_contract.functions.getMinPriorityFeePerGas().call()
                    network_info += f"📋 Fees from Policy Contract ({neox_config['policy_contract']})\n"
                    
                    max_priority_fee_per_gas = min_priority_fee
                    max_fee_per_gas = base_fee * 2 + max_priority_fee_per_gas
                except Exception as e:
                    network_info += f"⚠️  Policy Contract fetch failed: {e}, using fallback\n"
                    base_fee = w3.to_wei(1, "gwei")
                    max_priority_fee_per_gas = w3.to_wei(1, "gwei")
                    max_fee_per_gas = base_fee * 2 + max_priority_fee_per_gas
            else:
                # Standard EIP-1559: Get base fee from latest block
                latest = w3.eth.get_block("latest")
                base_fee = latest.get("baseFeePerGas", w3.to_wei(1, "gwei"))
                
                # Calculate fees
                max_priority_fee_per_gas = w3.to_wei(priority_gwei, "gwei")
                max_fee_per_gas = (
                    w3.to_wei(int(max_fee_gwei), "gwei")
                    if max_fee_gwei
                    else base_fee * 2 + max_priority_fee_per_gas
                )
            
            # Estimate gas if not provided
            if gas_limit:
                gas_limit = int(gas_limit)
            else:
                tx_for_estimate = {
                    "from": Web3.to_checksum_address(from_addr),
                    "to": Web3.to_checksum_address(to_addr) if to_addr else None,
                    "value": value_wei,
                    "data": data_hex,
                    "type": 2,
                    "maxFeePerGas": max_fee_per_gas,
                    "maxPriorityFeePerGas": max_priority_fee_per_gas,
                }
                try:
                    gas_limit = w3.eth.estimate_gas(tx_for_estimate)
                except Exception:
                    # Fallback gas estimation
                    code = w3.eth.get_code(Web3.to_checksum_address(to_addr)) if to_addr else b""
                    gas_limit = 21000 if (data_hex == "0x" and len(code) == 0) else 100000
            
            # Helper function for int to bytes
            def int_to_bytes(value: int) -> bytes:
                if value == 0:
                    return b""
                return value.to_bytes((value.bit_length() + 7) // 8, byteorder="big")
            
            # Build EIP-1559 transaction fields
            fields = [
                int_to_bytes(chain_id),
                int_to_bytes(nonce),
                int_to_bytes(max_priority_fee_per_gas),
                int_to_bytes(max_fee_per_gas),
                int_to_bytes(gas_limit),
                bytes.fromhex(to_addr[2:]) if to_addr else b"",
                int_to_bytes(value_wei),
                bytes.fromhex(data_hex[2:]) if data_hex and data_hex != "0x" else b"",
                [],  # accessList empty
            ]
            
            # Encode transaction
            encoded = rlp.encode(fields)
            unsigned_hex = "0x02" + encoded.hex()
            
            network_type = "NeoX (Policy-based)" if is_neox else "EIP-1559"
            
            return f"""✅ Unsigned Transaction Built ({network_type}):
                        {network_info}
                        Chain ID: {chain_id}
                        Nonce: {nonce}
                        Max Priority Fee: {max_priority_fee_per_gas} wei ({w3.from_wei(max_priority_fee_per_gas, 'gwei')} gwei)
                        Max Fee: {max_fee_per_gas} wei ({w3.from_wei(max_fee_per_gas, 'gwei')} gwei)
                        Gas Limit: {gas_limit}
                        To: {to_addr}
                        Value: {value_wei} wei
                        Data: {data_hex}
                        Unsigned Transaction: {unsigned_hex}
                        Use this with sign_evm_transaction tool to sign the transaction."""
            
        except ImportError:
            return "❌ web3 and rlp not installed. Install with: pip install web3 rlp"
        except Exception as e:
            return f"❌ Failed to build transaction: {str(e)}"


class ListAllAccountsTool(TurnkeyBaseTool):
    """List all accounts across all wallets in the organization"""
    
    name: str = "list_all_accounts"
    description: str = "List all wallet accounts within organization, returns comprehensive account information"
    parameters: dict = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "string",
                "description": "Max accounts per wallet (default: 50)"
            }
        },
        "required": []
    }
    
    async def execute(self, limit: str = "50", **kwargs) -> str:
        """List all accounts across all wallets"""
        try:
            accounts = []
            wallets_resp = self.client.list_wallets()
            
            for wallet in wallets_resp.get("wallets", []) or []:
                wallet_id = wallet.get("walletId")
                wallet_name = wallet.get("walletName", "")
                
                wa_resp = self.client.list_wallet_accounts(wallet_id, limit=limit)
                for acc in wa_resp.get("accounts", []) or []:
                    address = acc.get("address")
                    if address:
                        accounts.append({
                            "walletId": wallet_id,
                            "walletName": wallet_name,
                            "address": address,
                            "path": acc.get("path", "")
                        })
            
            if not accounts:
                return "ℹ️ No accounts found in the organization"
            
            account_list = []
            for i, acc in enumerate(accounts, 1):
                account_list.append(
                    f"{i}. {acc['walletName']} · {acc['address']} (Path: {acc['path']})"
                )
            
            return f"✅ Found {len(accounts)} accounts across {len(wallets_resp.get('wallets', []))} wallets:\n" + "\n".join(account_list)
            
        except Exception as e:
            return f"❌ Failed to list all accounts: {str(e)}"


class BatchSignTransactionsTool(TurnkeyBaseTool):
    """Batch sign transactions for multiple accounts"""
    
    name: str = "batch_sign_transactions"
    description: str = "Batch sign transactions for multiple accounts in the organization"
    parameters: dict = {
        "type": "object",
        "properties": {
            "to_address": {
                "type": "string",
                "description": "Recipient address (same for all transactions, required)"
            },
            "value_wei": {
                "type": "string",
                "description": "Value in wei per transaction (required)"
            },
            "data_hex": {
                "type": "string",
                "description": "Transaction data hex (default: 0x)"
            },
            "max_accounts": {
                "type": "string",
                "description": "Maximum number of accounts to process (default: 3)"
            },
            "enable_broadcast": {
                "type": "boolean",
                "description": "Whether to broadcast transactions (default: false)"
            },
            "rpc_url": {
                "type": "string",
                "description": "RPC URL (optional, uses WEB3_RPC_URL from environment)"
            }
        },
        "required": ["to_address", "value_wei"]
    }
    
    async def execute(self, to_address: str, value_wei: str, data_hex: str = "0x",
                     max_accounts: str = "3", enable_broadcast: bool = False,
                     rpc_url: str = None, **kwargs) -> str:
        """Batch sign transactions for multiple accounts"""
        try:
            from web3 import Web3
            import rlp
            
            # Get RPC URL
            rpc = rpc_url or os.getenv("WEB3_RPC_URL")
            if not rpc:
                return "❌ RPC URL not provided. Set WEB3_RPC_URL environment variable or provide rpc_url parameter."
            
            w3 = Web3(Web3.HTTPProvider(rpc))
            
            # Inject PoA middleware
            try:
                from web3.middleware import ExtraDataToPOAMiddleware
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except ImportError:
                from web3.middleware import geth_poa_middleware
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not w3.is_connected():
                return "❌ Cannot connect to RPC endpoint"
            
            # Get all accounts
            accounts = []
            wallets_resp = self.client.list_wallets()
            for wallet in wallets_resp.get("wallets", []) or []:
                wallet_id = wallet.get("walletId")
                wa_resp = self.client.list_wallet_accounts(wallet_id, limit="50")
                for acc in wa_resp.get("accounts", []) or []:
                    address = acc.get("address")
                    if address:
                        accounts.append(address)
            
            if not accounts:
                return "ℹ️ No accounts found in the organization"
            
            # Limit number of accounts
            max_acc = int(max_accounts)
            accounts = accounts[:max_acc]
            
            # Helper function
            def int_to_bytes(value: int) -> bytes:
                if value == 0:
                    return b""
                return value.to_bytes((value.bit_length() + 7) // 8, byteorder="big")
            
            # Build and sign transactions for each account
            results = []
            value_wei_int = int(value_wei)
            
            for from_addr in accounts:
                try:
                    # Get nonce
                    nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(from_addr))
                    
                    # Get gas parameters
                    try:
                        latest = w3.eth.get_block("latest")
                        base_fee = latest.get("baseFeePerGas", w3.to_wei(1, "gwei"))
                    except:
                        base_fee = w3.to_wei(1, "gwei")
                    
                    max_priority_fee_per_gas = w3.to_wei(1, "gwei")
                    max_fee_per_gas = base_fee * 2 + max_priority_fee_per_gas
                    
                    # Build unsigned transaction
                    fields = [
                        int_to_bytes(w3.eth.chain_id),
                        int_to_bytes(nonce),
                        int_to_bytes(max_priority_fee_per_gas),
                        int_to_bytes(max_fee_per_gas),
                        int_to_bytes(21000),
                        bytes.fromhex(to_address[2:]),
                        int_to_bytes(value_wei_int),
                        bytes.fromhex(data_hex[2:]) if data_hex != "0x" else b"",
                        [],
                    ]
                    
                    encoded = rlp.encode(fields)
                    unsigned_hex = "0x02" + encoded.hex()
                    
                    # Sign transaction
                    resp = self.client.sign_evm_transaction(
                        sign_with=from_addr,
                        unsigned_tx=unsigned_hex
                    )
                    
                    signed_tx = (
                        resp.get("activity", {})
                        .get("result", {})
                        .get("signTransactionResult", {})
                        .get("signedTransaction")
                    )
                    
                    if not signed_tx:
                        results.append(f"  • {from_addr[:10]}... ❌ Signing failed")
                        continue
                    
                    # Broadcast if enabled
                    if enable_broadcast and value_wei_int > 0:
                        try:
                            from eth_utils import to_bytes
                            tx_hash = w3.eth.send_raw_transaction(to_bytes(hexstr=signed_tx))
                            results.append(f"  • {from_addr[:10]}... ✅ Broadcasted: {tx_hash.hex()}")
                        except Exception as e:
                            results.append(f"  • {from_addr[:10]}... ❌ Broadcast failed: {e}")
                    else:
                        results.append(f"  • {from_addr[:10]}... ✅ Signed (not broadcasted)")
                        
                except Exception as e:
                    results.append(f"  • {from_addr[:10]}... ❌ Error: {e}")
            
            summary = f"✅ Batch signing completed for {len(accounts)} accounts:\n" + "\n".join(results)
            if not enable_broadcast:
                summary += f"\n\nℹ️  Broadcast disabled. Set enable_broadcast=true to broadcast transactions."
            
            return summary
            
        except Exception as e:
            return f"❌ Batch signing failed: {str(e)}"


class CreateWalletTool(TurnkeyBaseTool):
    """Create a new wallet"""
    
    name: str = "create_wallet"
    description: str = "Create a new wallet with specified accounts"
    parameters: dict = {
        "type": "object",
        "properties": {
            "wallet_name": {
                "type": "string",
                "description": "Wallet name (required)"
            },
            "accounts_json": {
                "type": "string",
                "description": "Accounts configuration as JSON string (optional, defaults to single Ethereum account). Example: '[{\"curve\":\"CURVE_SECP256K1\",\"pathFormat\":\"PATH_FORMAT_BIP32\",\"path\":\"m/44\\'/60\\'/0\\'/0/0\",\"addressFormat\":\"ADDRESS_FORMAT_ETHEREUM\"}]'"
            },
            "mnemonic_length": {
                "type": "string",
                "description": "Mnemonic length (12, 15, 18, 21, or 24, default: 24)"
            }
        },
        "required": ["wallet_name"]
    }
    
    async def execute(self, wallet_name: str, accounts_json: str = None, 
                     mnemonic_length: str = "24", **kwargs) -> str:
        """Create a new wallet"""
        try:
            # Parse accounts configuration
            if accounts_json:
                accounts = json.loads(accounts_json)
            else:
                # Default Ethereum account
                accounts = [{
                    "curve": "CURVE_SECP256K1",
                    "pathFormat": "PATH_FORMAT_BIP32",
                    "path": "m/44'/60'/0'/0/0",
                    "addressFormat": "ADDRESS_FORMAT_ETHEREUM"
                }]
            
            result = self.client.create_wallet(
                wallet_name=wallet_name,
                accounts=accounts,
                mnemonic_length=int(mnemonic_length)
            )
            
            wallet_id = result.get("activity", {}).get("result", {}).get("createWalletResult", {}).get("walletId")
            addresses = result.get("activity", {}).get("result", {}).get("createWalletResult", {}).get("addresses", [])
            
            if wallet_id:
                addr_list = "\n".join([f"  • {addr}" for addr in addresses])
                return f"✅ Wallet created successfully:\nWallet ID: {wallet_id}\nWallet Name: {wallet_name}\nAddresses:\n{addr_list}"
            else:
                return f"❌ Wallet creation failed:\n{json.dumps(result, indent=2)}"
                
        except Exception as e:
            return f"❌ Failed to create wallet: {str(e)}"


class GetWalletTool(TurnkeyBaseTool):
    """Get wallet information by wallet ID"""
    
    name: str = "get_wallet"
    description: str = "Get detailed information about a specific wallet"
    parameters: dict = {
        "type": "object",
        "properties": {
            "wallet_id": {
                "type": "string",
                "description": "Wallet ID (required)"
            }
        },
        "required": ["wallet_id"]
    }
    
    async def execute(self, wallet_id: str, **kwargs) -> str:
        """Get wallet information"""
        try:
            result = self.client.get_wallet(wallet_id)
            
            wallet = result.get("wallet", {})
            wallet_name = wallet.get("walletName", "Unknown")
            accounts = wallet.get("accounts", [])
            
            account_list = []
            for acc in accounts:
                account_list.append(f"  • {acc.get('address', 'Unknown')} (Path: {acc.get('path', 'Unknown')})")
            
            return f"✅ Wallet Information:\nWallet ID: {wallet_id}\nWallet Name: {wallet_name}\nAccounts ({len(accounts)}):\n" + "\n".join(account_list)
            
        except Exception as e:
            return f"❌ Failed to get wallet: {str(e)}"


class CreateWalletAccountsTool(TurnkeyBaseTool):
    """Add accounts to an existing wallet"""
    
    name: str = "create_wallet_accounts"
    description: str = "Add new accounts to an existing wallet"
    parameters: dict = {
        "type": "object",
        "properties": {
            "wallet_id": {
                "type": "string",
                "description": "Wallet ID (required)"
            },
            "accounts_json": {
                "type": "string",
                "description": "Accounts configuration as JSON string (required). Example: '[{\"curve\":\"CURVE_SECP256K1\",\"pathFormat\":\"PATH_FORMAT_BIP32\",\"path\":\"m/44\\'/60\\'/0\\'/0/1\",\"addressFormat\":\"ADDRESS_FORMAT_ETHEREUM\"}]'"
            }
        },
        "required": ["wallet_id", "accounts_json"]
    }
    
    async def execute(self, wallet_id: str, accounts_json: str, **kwargs) -> str:
        """Add accounts to existing wallet"""
        try:
            import json
            accounts = json.loads(accounts_json)
            
            result = self.client.create_wallet_accounts(
                wallet_id=wallet_id,
                accounts=accounts
            )
            
            addresses = result.get("activity", {}).get("result", {}).get("createWalletAccountsResult", {}).get("addresses", [])
            
            if addresses:
                address_list = "\n".join([f"• {addr}" for addr in addresses])
                return f"✅ Successfully added {len(addresses)} accounts to wallet {wallet_id}:\n{address_list}"
            else:
                return f"✅ Account creation initiated for wallet {wallet_id}"
                
        except json.JSONDecodeError:
            return "❌ Invalid JSON format for accounts_json parameter"
        except Exception as e:
            return f"❌ Failed to create wallet accounts: {str(e)}"


class CompleteTransactionWorkflowTool(TurnkeyBaseTool):
    """Complete transaction workflow: build, sign, and optionally broadcast"""
    
    name: str = "complete_transaction_workflow"
    description: str = "Complete transaction workflow: build unsigned tx, sign with Turnkey, and optionally broadcast"
    parameters: dict = {
        "type": "object",
        "properties": {
            "sign_with": {
                "type": "string",
                "description": "Signing identity (wallet account address, required)"
            },
            "to_address": {
                "type": "string",
                "description": "Recipient address (required)"
            },
            "value_wei": {
                "type": "string",
                "description": "Value in wei (required)"
            },
            "data_hex": {
                "type": "string",
                "description": "Transaction data hex (default: 0x)"
            },
            "enable_broadcast": {
                "type": "boolean",
                "description": "Whether to broadcast the transaction (default: true)"
            },
            "rpc_url": {
                "type": "string",
                "description": "RPC URL (optional, uses WEB3_RPC_URL from environment)"
            }
        },
        "required": ["sign_with", "to_address", "value_wei"]
    }
    
    async def execute(self, sign_with: str, to_address: str, value_wei: str,
                     data_hex: str = "0x", enable_broadcast: bool = False,
                     rpc_url: str = None, **kwargs) -> str:
        """Complete transaction workflow"""
        try:
            from web3 import Web3
            import rlp
            
            # Get RPC URL
            rpc = rpc_url or os.getenv("WEB3_RPC_URL")
            if not rpc:
                return "❌ RPC URL not provided"
            
            w3 = Web3(Web3.HTTPProvider(rpc))
            
            # Inject PoA middleware
            try:
                from web3.middleware import ExtraDataToPOAMiddleware
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except ImportError:
                from web3.middleware import geth_poa_middleware
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not w3.is_connected():
                return "❌ Cannot connect to RPC endpoint"
            
            # Helper function
            def int_to_bytes(value: int) -> bytes:
                if value == 0:
                    return b""
                return value.to_bytes((value.bit_length() + 7) // 8, byteorder="big")
            
            # Step 1: Build unsigned transaction
            chain_id = w3.eth.chain_id
            nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(sign_with))
            
            try:
                latest = w3.eth.get_block("latest")
                base_fee = latest.get("baseFeePerGas", w3.to_wei(1, "gwei"))
            except:
                base_fee = w3.to_wei(1, "gwei")
            
            max_priority_fee_per_gas = w3.to_wei(1, "gwei")
            max_fee_per_gas = base_fee * 2 + max_priority_fee_per_gas
            
            fields = [
                int_to_bytes(chain_id),
                int_to_bytes(nonce),
                int_to_bytes(max_priority_fee_per_gas),
                int_to_bytes(max_fee_per_gas),
                int_to_bytes(21000),
                bytes.fromhex(to_address[2:]),
                int_to_bytes(int(value_wei)),
                bytes.fromhex(data_hex[2:]) if data_hex != "0x" else b"",
                [],
            ]
            
            encoded = rlp.encode(fields)
            unsigned_hex = "0x02" + encoded.hex()
            
            workflow_log = []
            workflow_log.append(f"📋 Step 1: Built unsigned transaction")
            workflow_log.append(f"   Chain ID: {chain_id}")
            workflow_log.append(f"   From: {sign_with}")
            workflow_log.append(f"   To: {to_address}")
            workflow_log.append(f"   Value: {value_wei} wei")
            
            # Step 2: Sign transaction
            resp = self.client.sign_evm_transaction(
                sign_with=sign_with,
                unsigned_tx=unsigned_hex
            )
            
            signed_tx = (
                resp.get("activity", {})
                .get("result", {})
                .get("signTransactionResult", {})
                .get("signedTransaction")
            )
            
            if not signed_tx:
                return "❌ Transaction signing failed:\n" + "\n".join(workflow_log)
            
            workflow_log.append(f"✅ Step 2: Transaction signed successfully")
            
            # Step 3: Broadcast (optional)
            if enable_broadcast:
                try:
                    from eth_utils import to_bytes
                    tx_hash = w3.eth.send_raw_transaction(to_bytes(hexstr=signed_tx))
                    workflow_log.append(f"🚀 Step 3: Transaction broadcasted")
                    workflow_log.append(f"   TxHash: {tx_hash.hex()}")
                    workflow_log.append(f"   Explorer: https://sepolia.etherscan.io/tx/{tx_hash.hex()}")
                except Exception as e:
                    workflow_log.append(f"❌ Step 3: Broadcast failed: {e}")
            else:
                workflow_log.append(f"ℹ️  Step 3: Broadcast skipped (set enable_broadcast=true to broadcast)")
            
            return "\n".join(workflow_log)
            
        except Exception as e:
            return f"❌ Workflow failed: {str(e)}"


def get_turnkey_tools() -> List[BaseTool]:
    """Get all Turnkey tools"""
    return [
        # Signing tools
        SignEVMTransactionTool(),
        SignMessageTool(),
        SignTypedDataTool(),
        
        # Transaction tools
        BroadcastTransactionTool(),
        BuildUnsignedEIP1559TxTool(),
        CompleteTransactionWorkflowTool(),
        
        # Account management tools
        ListWalletsTool(),
        ListWalletAccountsTool(),
        ListAllAccountsTool(),
        GetWalletTool(),
        CreateWalletTool(),
        CreateWalletAccountsTool(),
        
        # Batch operation tools
        BatchSignTransactionsTool(),
        
        # Activity monitoring tools
        GetActivityTool(),
        ListActivitiesTool(),
        WhoAmITool(),
    ]
