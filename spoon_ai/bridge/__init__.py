"""
Cross-chain bridge module for DID synchronization
Ethereum ← → NeoFS/IPFS event indexing
"""

from .eth_neofs_indexer import EthereumNeoFSIndexer

__all__ = ["EthereumNeoFSIndexer"]


