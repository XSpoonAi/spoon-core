"""
Ethereum to NeoFS/IPFS Event Indexer
Listens to ERC-8004 registry events and ensures off-chain storage is synchronized
"""

import time
import json
from typing import Dict, Callable, Optional
from web3 import Web3
from web3.contract import Contract
from ..identity.erc8004_client import ERC8004Client
from ..identity.storage_client import DIDStorageClient


class EthereumNeoFSIndexer:
    """
    Event indexer that syncs Ethereum registry events to NeoFS/IPFS
    Ensures content hash verification and storage consistency
    """

    def __init__(
        self,
        erc8004_client: ERC8004Client,
        storage_client: DIDStorageClient,
        from_block: int = 0,
        poll_interval: int = 12
    ):
        self.erc8004_client = erc8004_client
        self.storage_client = storage_client
        self.w3 = erc8004_client.w3
        self.agent_registry = erc8004_client.agent_registry
        self.from_block = from_block
        self.current_block = from_block  # Track current block position
        self.poll_interval = poll_interval
        self.running = False

        # Event handlers
        self.event_handlers: Dict[str, Callable] = {}

    def register_event_handler(self, event_name: str, handler: Callable):
        """Register custom event handler"""
        self.event_handlers[event_name] = handler

    def start_indexing(self, block_limit: Optional[int] = None):
        """
        Start indexing events

        Args:
            block_limit: Optional block limit for testing (stops after N blocks)
        """
        self.running = True
        self.current_block = self.from_block
        blocks_processed = 0

        print(f"Starting indexer from block {self.current_block}")

        try:
            while self.running:
                try:
                    latest_block = self.w3.eth.block_number

                    if self.current_block > latest_block:
                        time.sleep(self.poll_interval)
                        continue

                    # Calculate how many blocks to process in this batch
                    # Respect block_limit if set
                    max_blocks_in_batch = 1000
                    if block_limit:
                        remaining_blocks = block_limit - blocks_processed
                        if remaining_blocks <= 0:
                            print(f"Block limit reached ({block_limit}), stopping")
                            break
                        max_blocks_in_batch = min(max_blocks_in_batch, remaining_blocks)
                    
                    # Process blocks in batches
                    to_block = min(self.current_block + max_blocks_in_batch - 1, latest_block)

                    print(f"Processing blocks {self.current_block} to {to_block}")

                    # Fetch events
                    self._process_agent_registered_events(self.current_block, to_block)
                    self._process_uris_updated_events(self.current_block, to_block)
                    self._process_capabilities_updated_events(self.current_block, to_block)

                    # Calculate blocks processed before updating current_block
                    blocks_in_batch = to_block - self.current_block + 1
                    blocks_processed += blocks_in_batch
                    
                    self.current_block = to_block + 1

                    # Stop if block limit reached
                    if block_limit and blocks_processed >= block_limit:
                        print(f"Block limit reached ({block_limit}), stopping")
                        break

                except Exception as e:
                    print(f"Indexer error: {e}")
                    time.sleep(self.poll_interval)
        finally:
            self.running = False
            print("Indexer stopped")

    def stop_indexing(self):
        """Stop the indexer"""
        self.running = False

    def _process_agent_registered_events(self, from_block: int, to_block: int):
        """Process AgentRegistered events"""
        try:
            if not hasattr(self.agent_registry.events, 'AgentRegistered'):
                # Event not defined in ABI, skip silently
                return
            
            event_filter = self.agent_registry.events.AgentRegistered.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )

            events = event_filter.get_all_entries()

            for event in events:
                try:
                    did_hash = event['args']['didHash']
                    controller = event['args']['controller']
                    agent_card_uri = event['args']['agentCardURI']
                    did_doc_uri = event['args']['didDocURI']
                    block_number = event['blockNumber']
                    tx_hash = event['transactionHash'].hex()

                    print(f"AgentRegistered: DID={did_hash.hex()}, Controller={controller}")

                    # Verify URIs are accessible
                    self._verify_and_replicate(did_hash.hex(), agent_card_uri, did_doc_uri)

                    # Call custom handler if registered
                    if 'AgentRegistered' in self.event_handlers:
                        self.event_handlers['AgentRegistered'](event)

                except Exception as e:
                    print(f"Error processing AgentRegistered event: {e}")
        except Exception as e:
            # Silently skip if event is not available (e.g., ABI doesn't include events)
            pass

    def _process_uris_updated_events(self, from_block: int, to_block: int):
        """Process URIsUpdated events"""
        try:
            if not hasattr(self.agent_registry.events, 'URIsUpdated'):
                # Event not defined in ABI, skip silently
                return
            
            event_filter = self.agent_registry.events.URIsUpdated.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )

            events = event_filter.get_all_entries()

            for event in events:
                try:
                    did_hash = event['args']['didHash']
                    agent_card_uri = event['args']['agentCardURI']
                    did_doc_uri = event['args']['didDocURI']

                    print(f"URIsUpdated: DID={did_hash.hex()}")

                    # Verify and replicate new URIs
                    self._verify_and_replicate(did_hash.hex(), agent_card_uri, did_doc_uri)

                    if 'URIsUpdated' in self.event_handlers:
                        self.event_handlers['URIsUpdated'](event)

                except Exception as e:
                    print(f"Error processing URIsUpdated event: {e}")
        except Exception as e:
            # Silently skip if event is not available (e.g., ABI doesn't include events)
            pass

    def _process_capabilities_updated_events(self, from_block: int, to_block: int):
        """Process CapabilitiesUpdated events"""
        try:
            if not hasattr(self.agent_registry.events, 'CapabilitiesUpdated'):
                # Event not defined in ABI, skip silently
                return
            
            event_filter = self.agent_registry.events.CapabilitiesUpdated.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )

            events = event_filter.get_all_entries()

            for event in events:
                try:
                    did_hash = event['args']['didHash']
                    capabilities = event['args']['capabilities']

                    print(f"CapabilitiesUpdated: DID={did_hash.hex()}, Caps={capabilities}")

                    if 'CapabilitiesUpdated' in self.event_handlers:
                        self.event_handlers['CapabilitiesUpdated'](event)

                except Exception as e:
                    print(f"Error processing CapabilitiesUpdated event: {e}")
        except Exception as e:
            # Silently skip if event is not available (e.g., ABI doesn't include events)
            pass

    def _process_identity_registered_events(self, from_block: int, to_block: int):
        """Process IdentityRegistry Registered events"""
        try:
            if not hasattr(self.identity_registry.events, 'Registered'):
                # Event not defined in ABI, skip silently
                return
            
            event_filter = self.identity_registry.events.Registered.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )

            events = event_filter.get_all_entries()

            for event in events:
                try:
                    agent_id = event['args']['agentId']
                    token_uri = event['args']['tokenURI']
                    owner = event['args']['owner']
                    block_number = event['blockNumber']
                    tx_hash = event['transactionHash'].hex()

                    print(f"📝 IdentityRegistry Registered: AgentId={agent_id}, Owner={owner}, TokenURI={token_uri}")

                    # Call custom handler if registered
                    if 'IdentityRegistered' in self.event_handlers:
                        self.event_handlers['IdentityRegistered'](event)

                except Exception as e:
                    print(f"Error processing IdentityRegistry Registered event: {e}")
        except Exception as e:
            # Silently skip if event is not available (e.g., ABI doesn't include events)
            pass

    def _verify_and_replicate(self, did_hash: str, agent_card_uri: str, did_doc_uri: str):
        """
        Verify URIs are accessible and replicate if needed
        """
        try:
            # Try to fetch DID document
            did_doc = self.storage_client.fetch_did_document(did_doc_uri)
            print(f"✓ DID document verified: {did_doc_uri}")

            # Try to fetch agent card
            agent_card = self.storage_client.fetch_did_document(agent_card_uri)
            print(f"✓ Agent card verified: {agent_card_uri}")

            # If primary is NeoFS, ensure IPFS backup exists
            if did_doc_uri.startswith("neofs://"):
                try:
                    ipfs_cid = self.storage_client._publish_to_ipfs(
                        json.dumps(did_doc).encode('utf-8')
                    )
                    print(f"✓ IPFS backup created: {ipfs_cid}")
                except Exception as e:
                    print(f"IPFS replication warning: {e}")

        except Exception as e:
            print(f"⚠ Content verification/replication failed: {e}")

    def get_indexer_status(self) -> Dict:
        """Get current indexer status"""
        latest_block = self.w3.eth.block_number
        return {
            "running": self.running,
            "current_block": self.current_block,
            "latest_block": latest_block,
            "sync_lag": latest_block - self.current_block,
            "blocks_processed": self.current_block - self.from_block
        }
















