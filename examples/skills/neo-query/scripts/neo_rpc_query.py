#!/usr/bin/env python3
"""
Comprehensive Neo RPC Query Script for neo-query skill.
Maps AI requests to the full set of tools in spoon_toolkits.crypto.neo.
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict

# Setup toolkit paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../"))
TOOLKIT_PATH = os.path.join(ROOT_DIR, "spoon-toolkit")

if TOOLKIT_PATH not in sys.path:
    sys.path.append(TOOLKIT_PATH)

try:
    from spoon_toolkits.crypto.neo.address_tools import GetAddressInfoTool, GetTransferByAddressTool
    from spoon_toolkits.crypto.neo.block_tools import GetBlockByHeightTool, GetBlockCountTool, GetBestBlockHashTool, GetRecentBlocksInfoTool
    from spoon_toolkits.crypto.neo.transaction_tools import GetRawTransactionByTransactionHashTool, GetTransactionCountByAddressTool
    from spoon_toolkits.crypto.neo.contract_tools import GetContractByHashTool, GetContractListByNameTool
    from spoon_toolkits.crypto.neo.nep_tools import GetNep17TransferByAddressTool, GetNep11TransferByAddressTool, GetNep17TransferCountByAddressTool
    from spoon_toolkits.crypto.neo.governance_tools import GetCommitteeInfoTool
    from spoon_toolkits.crypto.neo.voting_tools import GetCandidateCountTool, GetTotalVotesTool
    from spoon_toolkits.crypto.neo.log_state_tools import GetApplicationLogTool
    from spoon_toolkits.crypto.neo.asset_tools import GetAssetInfoByNameTool
except ImportError as e:
    print(json.dumps({
        "status": "error",
        "message": f"Failed to import neo toolkit: {str(e)}"
    }))
    sys.exit(1)


async def run_query(command: Dict[str, Any]) -> Dict[str, Any]:
    """Route the action to the appropriate toolkit tool."""
    action = command.get("action")
    network = command.get("network", "testnet")
    params = command.copy()
    params.pop("action", None)
    params.pop("network", None)
    
    try:
        if action == "get_address_info":
            tool = GetAddressInfoTool()
            result = await tool.execute(address=params.get("address"), network=network)
        
        elif action == "get_balance": # Alias for address info
            tool = GetAddressInfoTool()
            result = await tool.execute(address=params.get("address"), network=network)

        elif action == "get_block":
            height = params.get("height")
            if height is not None:
                tool = GetBlockByHeightTool()
                result = await tool.execute(block_height=int(height), network=network)
            else:
                tool = GetRecentBlocksInfoTool()
                result = await tool.execute(Limit=params.get("limit", 5), network=network)

        elif action == "get_transaction":
            tool = GetRawTransactionByTransactionHashTool()
            result = await tool.execute(tx_hash=params.get("hash"), network=network)

        elif action == "get_nep17_transfers":
            tool = GetNep17TransferByAddressTool()
            result = await tool.execute(
                address=params.get("address"), 
                Limit=params.get("limit", 10), 
                Skip=params.get("skip", 0), 
                network=network
            )

        elif action == "get_nep11_transfers":
            tool = GetNep11TransferByAddressTool()
            result = await tool.execute(
                address=params.get("address"), 
                Limit=params.get("limit", 10), 
                Skip=params.get("skip", 0), 
                network=network
            )

        elif action == "get_contract":
            tool = GetContractByHashTool()
            result = await tool.execute(contract_hash=params.get("hash"), network=network)

        elif action == "get_contract_list":
            tool = GetContractListByNameTool()
            result = await tool.execute(
                contract_name=params.get("name") or params.get("contract_name"), 
                Limit=params.get("limit", 10), 
                network=network
            )

        elif action == "get_committee":
            tool = GetCommitteeInfoTool()
            result = await tool.execute(network=network)

        elif action == "get_candidates":
            tool = GetCandidateCountTool()
            result = await tool.execute(network=network)

        elif action == "get_logs":
            tool = GetApplicationLogTool()
            result = await tool.execute(tx_hash=params.get("hash"), network=network)
            
        elif action == "get_asset_info":
            tool = GetAssetInfoByNameTool()
            result = await tool.execute(
                asset_name=params.get("name"),
                Limit=params.get("limit", 5),
                network=network
            )

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

        return {
            "status": "success",
            "action": action,
            "data": result.output if hasattr(result, "output") else str(result)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


async def main():
    """Main entry point."""
    try:
        input_text = sys.stdin.read().strip()
        if not input_text:
            print(json.dumps({"status": "error", "message": "No input provided"}))
            return
            
        command = json.loads(input_text)
    except json.JSONDecodeError:
        # Fallback: simple text input treated as balance check
        command = {"action": "get_address_info", "address": input_text}

    result = await run_query(command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
