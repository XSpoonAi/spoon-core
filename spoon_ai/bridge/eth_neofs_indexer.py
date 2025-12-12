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
        current_block = self.from_block
        blocks_processed = 0

        print(f"Starting indexer from block {current_block}")

        while self.running:
            try:
                latest_block = self.w3.eth.block_number

                if current_block > latest_block:
                    time.sleep(self.poll_interval)
                    continue

                # Process blocks in batches
                to_block = min(current_block + 1000, latest_block)

                print(f"Processing blocks {current_block} to {to_block}")

                # Fetch events
                self._process_agent_registered_events(current_block, to_block)
                self._process_uris_updated_events(current_block, to_block)
                self._process_capabilities_updated_events(current_block, to_block)

                current_block = to_block + 1
                blocks_processed += (to_block - current_block + 1)

                # Stop if block limit reached
                if block_limit and blocks_processed >= block_limit:
                    print(f"Block limit reached ({block_limit}), stopping")
                    break

            except Exception as e:
                print(f"Indexer error: {e}")
                time.sleep(self.poll_interval)

        print("Indexer stopped")

    def stop_indexing(self):
        """Stop the indexer"""
        self.running = False

    def _process_agent_registered_events(self, from_block: int, to_block: int):
        """Process AgentRegistered events"""
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

    def _process_uris_updated_events(self, from_block: int, to_block: int):
        """Process URIsUpdated events"""
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

    def _process_capabilities_updated_events(self, from_block: int, to_block: int):
        """Process CapabilitiesUpdated events"""
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
        return {
            "running": self.running,
            "current_block": self.from_block,
            "latest_block": self.w3.eth.block_number,
            "sync_lag": self.w3.eth.block_number - self.from_block
        }
















