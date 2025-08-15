import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from web3 import Web3, HTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware


class EthereumConfig:
    def __init__(self, rpc_url: str, chain_id: int, uniswap_router_address: str, uniswap_factory_address: str):
        self.rpc_url = rpc_url
        self.chain_id = chain_id
        self.uniswap_router_address = uniswap_router_address
        self.uniswap_factory_address = uniswap_factory_address
        
CHAINS = {
    "optimism": EthereumConfig(
        rpc_url="https://mainnet.optimism.io",
        chain_id=10,
        uniswap_router_address="0x1F98431c8aD98523631AE4a59f267346ea31F984",
        uniswap_factory_address="0xE592427A0AEce92De3Edee1F18E0157C05861564"
    )
}

logger = logging.getLogger(__name__)

class UniswapV3Client:
    """Uniswap V3 Trading Client"""
    
    # Uniswap V3 Router ABI (simplified version, only includes functions we need)
    ROUTER_ABI = json.loads('''
    [
        {
            "inputs": [
                {
                    "components": [
                        {"internalType": "address", "name": "tokenIn", "type": "address"},
                        {"internalType": "address", "name": "tokenOut", "type": "address"},
                        {"internalType": "uint24", "name": "fee", "type": "uint24"},
                        {"internalType": "address", "name": "recipient", "type": "address"},
                        {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
                        {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
                    ],
                    "internalType": "struct ISwapRouter.ExactInputSingleParams",
                    "name": "params",
                    "type": "tuple"
                }
            ],
            "name": "exactInputSingle",
            "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
            "stateMutability": "payable",
            "type": "function"
        },
        {
            "inputs": [
                {
                    "components": [
                        {"internalType": "bytes", "name": "path", "type": "bytes"},
                        {"internalType": "address", "name": "recipient", "type": "address"},
                        {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"}
                    ],
                    "internalType": "struct ISwapRouter.ExactInputParams",
                    "name": "params",
                    "type": "tuple"
                }
            ],
            "name": "exactInput",
            "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
            "stateMutability": "payable",
            "type": "function"
        }
    ]
    ''')
    
    # Uniswap V3 Factory ABI (simplified version)
    FACTORY_ABI = json.loads('''
    [
        {
            "inputs": [
                {"internalType": "address", "name": "tokenA", "type": "address"},
                {"internalType": "address", "name": "tokenB", "type": "address"},
                {"internalType": "uint24", "name": "fee", "type": "uint24"}
            ],
            "name": "getPool",
            "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    ''')
    
    # Uniswap V3 Pool ABI (simplified version)
    POOL_ABI = json.loads('''
    [
        {
            "inputs": [],
            "name": "slot0",
            "outputs": [
                {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                {"internalType": "int24", "name": "tick", "type": "int24"},
                {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
                {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
                {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
                {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
                {"internalType": "bool", "name": "unlocked", "type": "bool"}
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    ''')
    
    # ERC20 Token ABI (simplified version)
    ERC20_ABI = json.loads('''
    [
        {
            "inputs": [],
            "name": "decimals",
            "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "address", "name": "spender", "type": "address"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"}
            ],
            "name": "approve",
            "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    ''')
    
    # WETH ABI (simplified version, only includes deposit function)
    WETH_ABI = json.loads('''
    [
        {
            "constant": false,
            "inputs": [],
            "name": "deposit",
            "outputs": [],
            "payable": true,
            "stateMutability": "payable",
            "type": "function"
        },
        {
            "constant": false,
            "inputs": [{"name": "wad", "type": "uint256"}],
            "name": "withdraw",
            "outputs": [],
            "payable": false,
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "constant": true,
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "payable": false,
            "stateMutability": "view",
            "type": "function"
        },
        {
            "constant": true,
            "inputs": [{"name": "", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "payable": false,
            "stateMutability": "view",
            "type": "function"
        }
    ]
    ''')
    
    # Uniswap V2 Router ABI (simplified version, only includes functions we need)
    ROUTER_V2_ABI = json.loads('''
    [
        {
            "inputs": [
                {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                {"internalType": "address[]", "name": "path", "type": "address[]"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "deadline", "type": "uint256"}
            ],
            "name": "swapExactTokensForETH",
            "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                {"internalType": "address[]", "name": "path", "type": "address[]"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "deadline", "type": "uint256"}
            ],
            "name": "swapExactTokensForETHSupportingFeeOnTransferTokens",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                {"internalType": "address[]", "name": "path", "type": "address[]"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "deadline", "type": "uint256"}
            ],
            "name": "swapExactETHForTokens",
            "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
            "stateMutability": "payable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                {"internalType": "address[]", "name": "path", "type": "address[]"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "deadline", "type": "uint256"}
            ],
            "name": "swapExactTokensForTokens",
            "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "amountA", "type": "uint256"},
                {"internalType": "uint256", "name": "reserveA", "type": "uint256"},
                {"internalType": "uint256", "name": "reserveB", "type": "uint256"}
            ],
            "name": "getAmountOut",
            "outputs": [{"internalType": "uint256", "name": "amountB", "type": "uint256"}],
            "stateMutability": "pure",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                {"internalType": "address[]", "name": "path", "type": "address[]"}
            ],
            "name": "getAmountsOut",
            "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    ''')
    
    def __init__(self, chain_name: str = "optimism", private_key: Optional[str] = None):
        """
        Initialize Uniswap V3 client
        
        Args:
            chain_name: Chain name, default is "optimism"
            private_key: Private key for signing transactions
        """
        if chain_name not in CHAINS:
            raise ValueError(f"Unsupported chain: {chain_name}, available chains: {list(CHAINS.keys())}")
        
        self.config = CHAINS[chain_name]
        self.web3 = Web3(HTTPProvider(self.config.rpc_url))
        
        # Add POA middleware, suitable for testnets
        self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        # Initialize contracts
        self.router = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.config.uniswap_router_address),
            abi=self.ROUTER_ABI
        )
        
        # Initialize V2 router (if available)
        self.v2_router_addresses = {
            # Mainnet
            1: "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            # Sepolia
            11155111: "0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E",
            # Optimism
            10: "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
        }
        
        if self.config.chain_id in self.v2_router_addresses:
            self.router_v2 = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.v2_router_addresses[self.config.chain_id]),
                abi=self.ROUTER_V2_ABI
            )
        else:
            self.router_v2 = None
            logger.warning(f"Uniswap V2 router not available for chain ID {self.config.chain_id}")
        
        self.factory = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.config.uniswap_factory_address),
            abi=self.FACTORY_ABI
        )
        
        # Set up account
        self.private_key = private_key
        self.account = None
        if private_key:
            self.account = self.web3.eth.account.from_key(private_key)
            logger.info(f"Account set up: {self.account.address}")
    
    def get_pool_address(self, token_a: str, token_b: str, fee: int = 3000) -> str:
        """
        Get Uniswap V3 pool address
        
        Args:
            token_a: Token A address
            token_b: Token B address
            fee: Fee tier, default is 0.3% (3000)
            
        Returns:
            Pool address
        """
        token_a = self.web3.to_checksum_address(token_a)
        token_b = self.web3.to_checksum_address(token_b)
        
        # Ensure token_a < token_b (Uniswap requirement)
        if token_a.lower() > token_b.lower():
            token_a, token_b = token_b, token_a
            
        pool_address = self.factory.functions.getPool(token_a, token_b, fee).call()
        
        if pool_address == "0x0000000000000000000000000000000000000000":
            raise ValueError(f"Pool not found for token pair {token_a}/{token_b}")
            
        return pool_address
    
    def get_token_decimals(self, token_address: str) -> int:
        """Get token decimals"""
        token = self.web3.eth.contract(
            address=self.web3.to_checksum_address(token_address),
            abi=self.ERC20_ABI
        )
        return token.functions.decimals().call()
    
    def get_price(self, token_in: str, token_out: str, fee: int = 3000) -> float:
        """
        Get token price
        
        Args:
            token_in: Input token address
            token_out: Output token address
            fee: Fee tier, default is 0.3% (3000)
            
        Returns:
            Price (token_out/token_in)
        """
        token_in = self.web3.to_checksum_address(token_in)
        token_out = self.web3.to_checksum_address(token_out)
        
        # Get pool address
        pool_address = self.get_pool_address(token_in, token_out, fee)
        
        # Create pool contract
        pool = self.web3.eth.contract(address=pool_address, abi=self.POOL_ABI)
        
        # Get current price
        slot0 = pool.functions.slot0().call()
        sqrt_price_x96 = slot0[0]
        
        # Calculate price
        price = (sqrt_price_x96 / (2**96))**2
        
        # If token_in > token_out, take reciprocal
        if token_in.lower() > token_out.lower():
            price = 1 / price
            
        # Adjust for decimals
        decimals_in = self.get_token_decimals(token_in)
        decimals_out = self.get_token_decimals(token_out)
        price = price * (10 ** (decimals_in - decimals_out))
        
        return price
    
    def approve_token(self, token_address: str, amount: int) -> str:
        """
        Approve tokens for Uniswap router contract
        
        Args:
            token_address: Token address
            amount: Approval amount
            
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key not set, cannot execute transaction")
            
        token = self.web3.eth.contract(
            address=self.web3.to_checksum_address(token_address),
            abi=self.ERC20_ABI
        )
        
        # Build transaction
        tx = token.functions.approve(
            self.config.uniswap_router_address,
            amount
        ).build_transaction({
            'from': self.account.address,
            'gas': 200000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id
        })
        
        # Sign and send transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Approval transaction sent: {tx_hash.hex()}")
        return tx_hash.hex()
    
    def swap_exact_input_single(
        self, 
        token_in: str, 
        token_out: str, 
        amount_in: int, 
        amount_out_minimum: int = 0,
        fee: int = 3000,
        recipient: Optional[str] = None,
        deadline: Optional[int] = None
    ) -> str:
        """
        Execute exact input single token swap
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount
            amount_out_minimum: Minimum output amount, default is 0
            fee: Fee tier, default is 0.3% (3000)
            recipient: Recipient address, default is current account
            deadline: Transaction deadline timestamp, default is current time + 20 minutes
            
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key not set, cannot execute transaction")
            
        token_in = self.web3.to_checksum_address(token_in)
        token_out = self.web3.to_checksum_address(token_out)
        recipient = recipient or self.account.address
        recipient = self.web3.to_checksum_address(recipient)
        
        if deadline is None:
            deadline = self.web3.eth.get_block('latest')['timestamp'] + 1200  # 20 minutes
            
        # Build transaction parameters
        params = {
            'tokenIn': token_in,
            'tokenOut': token_out,
            'fee': fee,
            'recipient': recipient,
            'deadline': deadline,
            'amountIn': amount_in,
            'amountOutMinimum': amount_out_minimum,
            'sqrtPriceLimitX96': 0  # 0 means no limit
        }
        
        # Build transaction
        tx = self.router.functions.exactInputSingle(params).build_transaction({
            'from': self.account.address,
            'gas': 500000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id,
            'value': 0
        })
        
        # Sign and send transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Swap transaction sent: {tx_hash.hex()}")
        return tx_hash.hex()
    
    def get_token_balance(self, token_address: str, address: Optional[str] = None) -> int:
        """
        Get token balance
        
        Args:
            token_address: Token address
            address: Query address, default is current account
            
        Returns:
            Token balance
        """
        if not address and not self.account:
            raise ValueError("Account address not set")
            
        address = address or self.account.address
        token = self.web3.eth.contract(
            address=self.web3.to_checksum_address(token_address),
            abi=self.ERC20_ABI
        )
        
        return token.functions.balanceOf(address).call()
    
    def get_token_allowance(self, token_address: str, spender: Optional[str] = None) -> int:
        """
        Get token allowance
        
        Args:
            token_address: Token address
            spender: Authorized address, defaults to Uniswap router contract
            
        Returns:
            Allowance amount
        """
        if not self.account:
            raise ValueError("Account address not set")
            
        spender = spender or self.config.uniswap_router_address
        token = self.web3.eth.contract(
            address=self.web3.to_checksum_address(token_address),
            abi=self.ERC20_ABI + [
                {
                    "constant": True,
                    "inputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"}
                    ],
                    "name": "allowance",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "payable": False,
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        )
        
        return token.functions.allowance(self.account.address, spender).call()
    
    def get_eth_balance(self, address: Optional[str] = None) -> int:
        """
        Get ETH balance
        
        Args:
            address: Address to check balance for, default is current account
            
        Returns:
            ETH balance in wei
        """
        if not address and not self.account:
            raise ValueError("Account address not set")
            
        address = address or self.account.address
        return self.web3.eth.get_balance(address)
    
    def wrap_eth(self, amount: int) -> str:
        """
        Convert ETH to WETH by depositing ETH into the WETH contract
        
        Args:
            amount: Amount of ETH to wrap (in wei)
            
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key not set, cannot execute transaction")
        
        # WETH contract addresses for different networks
        weth_addresses = {
            # Mainnet WETH
            1: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            # Sepolia WETH
            11155111: "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9",
            # Optimism WETH
            10: "0x4200000000000000000000000000000000000006"
        }
        
        if self.config.chain_id not in weth_addresses:
            raise ValueError(f"WETH address not configured for chain ID {self.config.chain_id}")
        
        weth_address = weth_addresses[self.config.chain_id]
        weth_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(weth_address),
            abi=self.WETH_ABI
        )
        
        # Build transaction
        tx = weth_contract.functions.deposit().build_transaction({
            'from': self.account.address,
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id,
            'value': amount
        })
        
        # Sign and send transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"ETH to WETH conversion transaction sent: {tx_hash.hex()}")
        return tx_hash.hex()
    
    def unwrap_eth(self, amount: int) -> str:
        """
        Convert WETH back to ETH by withdrawing from the WETH contract
        
        Args:
            amount: Amount of WETH to unwrap (in wei)
            
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key not set, cannot execute transaction")
        
        # WETH contract addresses for different networks
        weth_addresses = {
            # Mainnet WETH
            1: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            # Sepolia WETH
            11155111: "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9",
            # Optimism WETH
            10: "0x4200000000000000000000000000000000000006"
        }
        
        if self.config.chain_id not in weth_addresses:
            raise ValueError(f"WETH address not configured for chain ID {self.config.chain_id}")
        
        weth_address = weth_addresses[self.config.chain_id]
        weth_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(weth_address),
            abi=self.WETH_ABI
        )
        
        # Build transaction
        tx = weth_contract.functions.withdraw(amount).build_transaction({
            'from': self.account.address,
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id
        })
        
        # Sign and send transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"WETH to ETH conversion transaction sent: {tx_hash.hex()}")
        return tx_hash.hex()
    
    def get_weth_address(self) -> str:
        """
        Get the WETH contract address for the current chain
        
        Returns:
            WETH contract address
        """
        weth_addresses = {
            # Mainnet WETH
            1: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            # Sepolia WETH
            11155111: "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9",
            # Optimism WETH
            10: "0x4200000000000000000000000000000000000006"
        }
        
        if self.config.chain_id not in weth_addresses:
            raise ValueError(f"WETH address not configured for chain ID {self.config.chain_id}")
        
        return weth_addresses[self.config.chain_id]
    
    def get_v2_amounts_out(self, amount_in: int, path: list) -> list:
        """
        Get expected output amounts for a given input amount and path using Uniswap V2
        
        Args:
            amount_in: Input amount
            path: Array of token addresses representing the swap path
            
        Returns:
            Array of output amounts for each step in the path
        """
        if not self.router_v2:
            raise ValueError("Uniswap V2 router not available for this chain")
            
        # Convert addresses to checksum format
        path = [self.web3.to_checksum_address(addr) for addr in path]
        # Call getAmountsOut function
        return self.router_v2.functions.getAmountsOut(amount_in, path).call()
    
    def swap_exact_tokens_for_eth(
        self,
        token_address: str,
        amount_in: int,
        amount_out_min: int = 0,
        recipient: Optional[str] = None,
        deadline: Optional[int] = None,
        supports_fee: bool = False
    ) -> str:
        """
        Swap exact amount of tokens for ETH using Uniswap V2
        
        Args:
            token_address: Address of the token to swap from
            amount_in: Exact amount of input tokens
            amount_out_min: Minimum amount of ETH to receive
            recipient: Recipient address for the ETH, default is current account
            deadline: Transaction deadline timestamp, default is current time + 20 minutes
            supports_fee: Whether to use the function that supports tokens with transfer fees
            
        Returns:
            Transaction hash
        """
        if not self.router_v2:
            raise ValueError("Uniswap V2 router not available for this chain")
            
        if not self.account:
            raise ValueError("Private key not set, cannot execute transaction")
            
        token_address = self.web3.to_checksum_address(token_address)
        recipient = recipient or self.account.address
        recipient = self.web3.to_checksum_address(recipient)
        
        if deadline is None:
            deadline = self.web3.eth.get_block('latest')['timestamp'] + 1200  # 20 minutes
            
        # Get WETH address
        weth_address = self.get_weth_address()
        
        # Create path: token -> WETH
        path = [token_address, weth_address]
        
        # Check if we have enough allowance
        token = self.web3.eth.contract(
            address=token_address,
            abi=self.ERC20_ABI
        )
        
        allowance = token.functions.allowance(
            self.account.address,
            self.v2_router_addresses[self.config.chain_id]
        ).call()
        
        if allowance < amount_in:
            logger.warning(f"Insufficient allowance: {allowance} < {amount_in}")
            logger.info("Approving tokens for Uniswap V2 router...")
            
            # Approve tokens
            approve_tx = token.functions.approve(
                self.v2_router_addresses[self.config.chain_id],
                amount_in
            ).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'chainId': self.config.chain_id
            })
            
            signed_tx = self.web3.eth.account.sign_transaction(approve_tx, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Approval transaction sent: {tx_hash.hex()}")
            
            # Wait for the approval transaction to be mined
            logger.info("Waiting for approval transaction to be mined...")
            timeout = getattr(self, 'tx_timeout', 120)  # configurable, default 120s
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            if receipt.status != 1:
                raise RuntimeError(f"Approval transaction failed. Receipt status: {receipt.status}")
        
        # Build swap transaction
        if supports_fee:
            # Use the function that supports tokens with transfer fees
            swap_func = self.router_v2.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
                amount_in,
                amount_out_min,
                path,
                recipient,
                deadline
            )
        else:
            # Use the standard function
            swap_func = self.router_v2.functions.swapExactTokensForETH(
                amount_in,
                amount_out_min,
                path,
                recipient,
                deadline
            )
        
        tx = swap_func.build_transaction({
            'from': self.account.address,
            'gas': 500000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id
        })
        
        # Sign and send transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Swap tokens for ETH transaction sent: {tx_hash.hex()}")
        # Wait for swap transaction receipt with timeout
        timeout = getattr(self, 'tx_timeout', 120)  # configurable, default 120s
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        if receipt.status != 1:
            raise RuntimeError(f"Swap transaction failed. Receipt status: {receipt.status}")
        logger.info(f"Swap tokens for ETH transaction successful: {tx_hash.hex()}")
        return tx_hash.hex()
    
    def swap_exact_eth_for_tokens(
        self,
        token_address: str,
        amount_in: int,
        amount_out_min: int = 0,
        recipient: Optional[str] = None,
        deadline: Optional[int] = None
    ) -> str:
        """
        Swap exact amount of ETH for tokens using Uniswap V2
        
        Args:
            token_address: Address of the token to swap to
            amount_in: Exact amount of ETH to send (in wei)
            amount_out_min: Minimum amount of tokens to receive
            recipient: Recipient address for the tokens, default is current account
            deadline: Transaction deadline timestamp, default is current time + 20 minutes
            
        Returns:
            Transaction hash
        """
        if not self.router_v2:
            raise ValueError("Uniswap V2 router not available for this chain")
            
        if not self.account:
            raise ValueError("Private key not set, cannot execute transaction")
            
        token_address = self.web3.to_checksum_address(token_address)
        recipient = recipient or self.account.address
        recipient = self.web3.to_checksum_address(recipient)
        
        if deadline is None:
            deadline = self.web3.eth.get_block('latest')['timestamp'] + 1200  # 20 minutes
            
        # Get WETH address
        weth_address = self.get_weth_address()
        
        # Create path: WETH -> token
        path = [weth_address, token_address]
        
        # Verify the token contract exists
        try:
            token_code = self.web3.eth.get_code(token_address)
            if token_code == b'' or token_code == '0x':
                logger.error(f"Token contract at {token_address} does not exist or is not deployed")
                raise ValueError(f"Token contract at {token_address} does not exist or is not deployed")
        except Exception as e:
            logger.error(f"Error checking token contract: {str(e)}")
            raise
            
        # Verify the router contract exists
        try:
            router_address = self.v2_router_addresses[self.config.chain_id]
            router_code = self.web3.eth.get_code(router_address)
            if router_code == b'' or router_code == '0x':
                logger.error(f"Router contract at {router_address} does not exist or is not deployed")
                raise ValueError(f"Router contract at {router_address} does not exist or is not deployed")
        except Exception as e:
            logger.error(f"Error checking router contract: {str(e)}")
            raise
            
        # Check if the pool exists
        try:
            # Try to get expected amounts to verify the pool exists
            amounts = self.get_v2_amounts_out(amount_in, path)
            logger.info(f"Expected output amounts: {amounts}")
        except Exception as e:
            logger.error(f"Error checking pool existence: {str(e)}")
            logger.error("This likely means the pool does not exist on this network")
            raise ValueError(f"Pool does not exist for path {path}: {str(e)}")
        
        # Build swap transaction
        try:
            tx = self.router_v2.functions.swapExactETHForTokens(
                amount_out_min,
                path,
                recipient,
                deadline
            ).build_transaction({
                'from': self.account.address,
                'gas': 500000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'chainId': self.config.chain_id,
                'value': amount_in
            })
        except Exception as e:
            logger.error(f"Error building transaction: {str(e)}")
            raise
        
        # Sign and send transaction
        try:
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Swap ETH for tokens transaction sent: {tx_hash.hex()}")
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"Error sending transaction: {str(e)}")
            raise
