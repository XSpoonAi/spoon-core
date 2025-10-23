import os
from dotenv import load_dotenv


def int_to_bytes(value: int) -> bytes:
    if value == 0:
        return b""
    return value.to_bytes((value.bit_length() + 7) // 8, byteorder="big")


def main():
    load_dotenv()
    try:
        from web3 import Web3
        import rlp
    except Exception as e:
        print("Please install required packages: pip install web3 rlp")
        raise

    rpc = os.getenv("WEB3_RPC_URL")
    from_addr = os.getenv("TURNKEY_SIGN_WITH")
    to_addr = os.getenv("TX_TO_ADDRESS", from_addr)
    value_wei = int(os.getenv("TX_VALUE_WEI", "0"))
    data_hex = os.getenv("TX_DATA_HEX", "0x")
    priority_gwei = int(os.getenv("TX_PRIORITY_FEE_GWEI", "1"))
    max_fee_gwei_override = os.getenv("TX_MAX_FEE_GWEI")

    if not rpc:
        print("WEB3_RPC_URL is required")
        return
    if not from_addr:
        print("TURNKEY_SIGN_WITH (from address) is required")
        return

    w3 = Web3(Web3.HTTPProvider(rpc))

    chain_id = w3.eth.chain_id
    nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(from_addr))
    
    # Get base fee from latest block
    try:
        latest = w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", w3.to_wei(1, "gwei"))
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch latest block: {e}")
        print("   Using default base fee of 1 gwei")
        base_fee = w3.to_wei(1, "gwei")
    max_priority_fee_per_gas = w3.to_wei(priority_gwei, "gwei")
    max_fee_per_gas = (
        w3.to_wei(int(max_fee_gwei_override), "gwei")
        if max_fee_gwei_override
        else base_fee * 2 + max_priority_fee_per_gas
    )

    tx_for_estimate = {
        "from": Web3.to_checksum_address(from_addr),
        "to": Web3.to_checksum_address(to_addr) if to_addr else None,
        "value": value_wei,
        "data": data_hex,
        "type": 2,
        "maxFeePerGas": max_fee_per_gas,
        "maxPriorityFeePerGas": max_priority_fee_per_gas,
    }
    # Optional explicit gas limit override
    gas_override = os.getenv("TX_GAS_LIMIT")
    if gas_override:
        gas_limit = int(gas_override)
    else:
        try:
            gas_limit = w3.eth.estimate_gas(tx_for_estimate)
        except Exception:
            # Heuristic fallback if node returns "gas required exceeds allowance (0)" or similar
            code = w3.eth.get_code(Web3.to_checksum_address(to_addr)) if to_addr else b""
            gas_limit = 21000 if (data_hex == "0x" and len(code) == 0) else 100000

    # Optional balance sanity check (may help diagnose estimate failures on some RPCs)
    try:
        balance = w3.eth.get_balance(Web3.to_checksum_address(from_addr))
        total_fee_worst = max_fee_per_gas * gas_limit + value_wei
        if balance < total_fee_worst:
            print("[warn] balance is lower than (value + gas*maxFeePerGas); funding may be required to broadcast.")
    except Exception:
        pass

    # EIP-1559 unsigned payload fields (list order is critical)
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

    encoded = rlp.encode(fields)
    unsigned_hex = "0x02" + encoded.hex()

    print("chainId:", chain_id)
    print("nonce:", nonce)
    print("maxPriorityFeePerGas:", max_priority_fee_per_gas)
    print("maxFeePerGas:", max_fee_per_gas)
    print("gasLimit:", gas_limit)
    print("to:", to_addr)
    print("value:", value_wei)
    print("data:", data_hex)
    print("TURNKEY_UNSIGNED_TX_HEX:")
    print(unsigned_hex)


if __name__ == "__main__":
    main()


