"""
Shared ERC-8004 ABI fragments (minimal, artifact-free).

These ABIs cover the common calls used by the Python SDK and demos.
"""

# Identity Registry
IDENTITY_ABI_MIN = [
    {
        "inputs": [],
        "name": "totalAgents",
        "outputs": [{"internalType": "uint256", "name": "count", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "tokenURI",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "string", "name": "key", "type": "string"},
        ],
        "name": "getMetadata",
        "outputs": [{"internalType": "bytes", "name": "value", "type": "bytes"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"indexed": False, "internalType": "string", "name": "tokenURI", "type": "string"},
            {"indexed": False, "internalType": "address", "name": "owner", "type": "address"},
        ],
        "name": "Registered",
        "type": "event",
    },
]

IDENTITY_ABI_WITH_REGISTER = IDENTITY_ABI_MIN + [
    {
        "inputs": [{"internalType": "string", "name": "tokenURI_", "type": "string"}],
        "name": "register",
        "outputs": [{"internalType": "uint256", "name": "agentId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

IDENTITY_ABI_WITH_METADATA_WRITE = IDENTITY_ABI_WITH_REGISTER + [
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "string", "name": "key", "type": "string"},
            {"internalType": "bytes", "name": "value", "type": "bytes"},
        ],
        "name": "setMetadata",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

# Reputation Registry (minimal)
REPUTATION_ABI_MIN = [
    {
        "inputs": [
            {"type": "uint256", "name": "agentId"},
            {"type": "uint8", "name": "score"},
            {"type": "bytes32", "name": "tag"},
            {"type": "bytes32", "name": "stage"},
            {"type": "string", "name": "uri"},
            {"type": "bytes32", "name": "paymentHash"},
            {"type": "bytes", "name": "feedbackAuth"},
        ],
        "name": "giveFeedback",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"type": "uint256", "name": "agentId"},
            {"type": "address", "name": "validator"},
            {"type": "uint64", "name": "index"},
        ],
        "name": "revokeFeedback",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"type": "uint256", "name": "agentId"},
            {"type": "address[]", "name": "validators"},
            {"type": "bytes32", "name": "tag"},
            {"type": "bytes32", "name": "stage"},
        ],
        "name": "getSummary",
        "outputs": [{"type": "uint64", "name": "score"}, {"type": "uint8", "name": "count"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Validation Registry (minimal)
VALIDATION_ABI_MIN = [
    {
        "inputs": [
            {"type": "address", "name": "validator"},
            {"type": "uint256", "name": "agentId"},
            {"type": "string", "name": "uri"},
            {"type": "bytes32", "name": "requestHash"},
        ],
        "name": "validationRequest",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"type": "bytes32", "name": "requestHash"},
            {"type": "uint8", "name": "score"},
            {"type": "string", "name": "uri"},
            {"type": "bytes32", "name": "paymentHash"},
            {"type": "bytes32", "name": "responseHash"},
        ],
        "name": "validationResponse",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"type": "bytes32", "name": "requestHash"}],
        "name": "getValidationStatus",
        "outputs": [
            {"type": "address", "name": "validator"},
            {"type": "uint256", "name": "agentId"},
            {"type": "uint8", "name": "score"},
            {"type": "bytes32", "name": "responseHash"},
            {"type": "uint256", "name": "timestamp"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

# Agent Registry (ChaosChain SpoonAgentRegistry minimal)
AGENT_REGISTRY_ABI = [
    {
        "inputs": [{"type": "bytes32"}, {"type": "string"}, {"type": "string"}, {"type": "bytes"}],
        "name": "registerAgent",
        "outputs": [{"type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"type": "bytes32"}],
        "name": "resolveAgent",
        "outputs": [
            {
                "components": [
                    {"type": "address[]"},
                    {"type": "string"},
                    {"type": "string"},
                    {"type": "string[]"},
                    {"type": "uint256"},
                    {"type": "bool"},
                ],
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"type": "bytes32"}, {"type": "string[]"}],
        "name": "updateCapabilities",
        "outputs": [{"type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


def get_abi(contract_name: str):
    mapping = {
        "ERC8004IdentityRegistry": IDENTITY_ABI_WITH_METADATA_WRITE,
        "ERC8004ReputationRegistry": REPUTATION_ABI_MIN,
        "ERC8004ValidationRegistry": VALIDATION_ABI_MIN,
        "SpoonAgentRegistry": AGENT_REGISTRY_ABI,
    }
    return mapping.get(contract_name)
