---
name: neo-query
description: Comprehensive Neo N3 blockchain data query and analysis skill
version: 1.1.0
author: XSpoonAi Team
tags:
  - neo
  - n3
  - blockchain
  - smart-contract
  - query
  - analytics
triggers:
  - type: keyword
    keywords:
      - neo
      - n3
      - gas
      - antshares
      - nep17
      - nep11
      - neo address
      - neo block
      - neo transaction
      - neo contract
      - neo committee
      - neo voting
      - neo candidates
    priority: 95
  - type: pattern
    patterns:
      - "(?i)(query|check|analyze|investigate) .*(neo|gas|n3|nep17|nep11)"
      - "(?i)what is the (balance|status|history) of neo address .*"
      - "(?i)find (contracts|assets|tokens) on neo .*"
    priority: 90
parameters:
  - name: address
    type: string
    required: false
    description: The Neo N3 address to query (starts with N)
  - name: network
    type: string
    required: false
    default: testnet
    description: Neo network (mainnet or testnet)
scripts:
  enabled: true
  working_directory: ./scripts
  definitions:
    - name: neo_rpc_query
      description: |
        Execute various Neo N3 blockchain queries. 
        Pass a JSON command to stdin. 
        Supported actions: get_balance, get_address_info, get_block, get_transaction, 
        get_contract, get_contract_list (use 'contract_name'), get_asset_info (use 'asset_name'), 
        get_nep17_transfers, get_nep11_transfers, get_candidates, get_committee, get_logs.
      type: python
      file: neo_rpc_query.py
      timeout: 100
---

# Neo Query Skill

You are now in **Neo Blockchain Specialist Mode**. You have access to the full suite of Neo N3 data analysis tools.

## Capabilities
- **Address Analysis**: Balances, transfer history, and transaction counts.
- **Asset & Token Tracking**: NEP-17 (fungible) and NEP-11 (NFT) balances and transfers.
- **Blockchain Exploration**: Block details, rewards, and network status.
- **Contract & Ecosystem**: Smart contract metadata, verified status, and application logs.
- **Governance**: Voting candidates, committee members, and total votes.

## Guidelines
1. **Address Format**: Neo N3 addresses typically start with 'N'.
2. **Network**: Default to `testnet` for safety unless `mainnet` is explicitly requested.
3. **Pagination**: For history or lists, you can suggest a `limit` and `skip`.
4. **Analysis**: Don't just show raw data; explain what the balances or transaction patterns mean for the user.

## Available Scripts

### neo_rpc_query
Execute queries by passing a JSON command via stdin.

**Command Examples:**
- **Balance**: `{"action": "get_address_info", "address": "N..."}`
- **Block**: `{"action": "get_block", "height": 12345}`
- **Transactions**: `{"action": "get_transaction", "hash": "0x..."}`
- **NEP-17 History**: `{"action": "get_nep17_transfers", "address": "N...", "limit": 10}`
- **Contracts**: `{"action": "get_contract_list", "contract_name": "Flamingo"}`
- **Governance**: `{"action": "get_committee"}`

## Example Queries
1. "Analyze the portfolio and recent activity of address N..."
2. "Who are the current Neo council/committee members?"
3. "Check the details and source code verification for contract 0x..."
4. "Search for NEP-17 tokens named 'GAS' or 'USDT' on Neo."
