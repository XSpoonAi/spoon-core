import os
import json
from dotenv import load_dotenv


def main():
    load_dotenv()

    print("🔐 Turnkey SDK Demo · Secure Signing & Audit Trail")
    print("=" * 50)
    print("This guided demo covers:")
    print("  1) EVM transaction signing (no local private keys)")
    print("  2) Message & EIP-712 signing (auth/orders)")
    print("  3) Optional on-chain broadcast + confirmation")
    print("  4) Activity history & audit trail")
    print()

    from spoon_ai.turnkey import Turnkey

    try:
        client = Turnkey()
        print("✅ Turnkey client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Turnkey client: {e}")
        print("💡 Fix tips:")
        print("   - Ensure .env contains: TURNKEY_API_PUBLIC_KEY, TURNKEY_API_PRIVATE_KEY, TURNKEY_ORG_ID")
        print("   - Verify keys belong to the same organization")
        return

    sign_with = os.getenv("TURNKEY_SIGN_WITH")
    if not sign_with:
        print("❌ TURNKEY_SIGN_WITH is not set.")
        print("💡 Set your wallet address or private key ID in .env (TURNKEY_SIGN_WITH)")
        return
        
    print(f"🔑 Using signing identity: {sign_with}")
    print()

    # Step result flags for summary
    tx_sign_ok = False
    broadcast_ok = False
    msg_sign_ok = False
    typed_ok = False
    audit_ok = False

    # 1) EVM Transaction Signing
    print("🧩 Step 1/4 · EVM Transaction Signing")
    print("-" * 40)
    unsigned_tx = os.getenv("TURNKEY_UNSIGNED_TX_HEX")
    if unsigned_tx:
        print(f"📄 Unsigned tx detected: {unsigned_tx[:50]}...")
        print("🔐 Requesting Turnkey to sign...")
        
        try:
            resp = client.sign_evm_transaction(sign_with=sign_with, unsigned_tx=unsigned_tx)
            tx_sign_ok = True
            print("✅ EVM transaction signed")
            print("📋 Signed Transaction (raw response):")
            print(json.dumps(resp, indent=2))
        except Exception as e:
            print(f"❌ Transaction signing failed: {e}")
            print("💡 Fix tips:")
            print("   - Ensure TURNKEY_SIGN_WITH is authorized to sign")
            print("   - Verify TURNKEY_UNSIGNED_TX_HEX is a valid EIP-1559 payload (0x02...) and chainId matches policy")
            print("   - Check Turnkey policy limits (amount/chain/targets)")
            return

        activity_id = (
            resp.get("activity", {}).get("id")
            or resp.get("activity", {}).get("result", {}).get("activity", {}).get("id")
        )
        if activity_id:
            detail = client.get_activity(activity_id)
            print("📜 Activity detail:")
            print(json.dumps(detail, indent=2))

        # Optional broadcast if RPC is configured and web3 is available
        rpc_url = os.getenv("WEB3_RPC_URL")
        if rpc_url:
            try:
                from web3 import Web3
                from eth_utils import to_bytes

                w3 = Web3(Web3.HTTPProvider(rpc_url))
                
                signed_tx = (
                    resp.get("activity", {})
                    .get("result", {})
                    .get("signTransactionResult", {})
                    .get("signedTransaction")
                )
                if signed_tx:
                    print(f"\n🚀 Broadcasting transaction...")
                    tx_hash = w3.eth.send_raw_transaction(to_bytes(hexstr=signed_tx))
                    tx_hash_hex = tx_hash.hex()
                    broadcast_ok = True
                    print(f"✅ Transaction broadcasted")
                    print(f"📝 TxHash: {tx_hash_hex}")
                    
                    # Determine network for explorer URL
                    chain_id = w3.eth.chain_id
                    if chain_id == 1:
                        explorer_url = f"https://etherscan.io/tx/{tx_hash_hex}"
                    elif chain_id == 11155111:  # Sepolia
                        explorer_url = f"https://sepolia.etherscan.io/tx/{tx_hash_hex}"
                    elif chain_id == 5:  # Goerli
                        explorer_url = f"https://goerli.etherscan.io/tx/{tx_hash_hex}"
                    else:
                        explorer_url = f"Chain {chain_id} - TxHash: {tx_hash_hex}"
                    
                    print(f"🔍 View on Explorer: {explorer_url}")
                    print(f"⏳ Waiting for confirmation...")
                    
                    try:
                        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                        print(f"✅ Transaction confirmed in block {receipt.blockNumber}")
                        if receipt.status == 1:
                            print("🎉 Transaction executed successfully!")
                        else:
                            print("❌ Transaction failed (reverted)")
                    except Exception as e:
                        print(f"⏳ Confirmation timeout (tx may still be pending): {e}")
                        
            except ImportError:
                print("📦 web3 not installed; skipping broadcast. Install with: pip install web3 eth-utils")
            except Exception as e:
                print(f"❌ Broadcast failed: {e}")
                print("💡 Fix tips:")
                print("   - Check WEB3_RPC_URL and network availability")
                print("   - Ensure account has enough ETH for gas on the target chain")
                print("   - If using testnets, confirm the RPC supports the network")
        else:
            print("ℹ️  Set WEB3_RPC_URL to enable automatic broadcasting")
    else:
        print("ℹ️ No unsigned tx configured (TURNKEY_UNSIGNED_TX_HEX is empty)")
        print("   Quick start:")
        print("   1) python -m examples.turnkey.build_unsigned_eip1559_tx")
        print("   2) Copy TURNKEY_UNSIGNED_TX_HEX into your .env")
    
    print()

    # 2) Message Signing
    print("🧩 Step 2/4 · Message Signing")
    print("-" * 40)
    msg = os.getenv("TURNKEY_SIGN_MESSAGE") or "hello turnkey"
    print(f"📝 Message to sign: '{msg}' (set TURNKEY_SIGN_MESSAGE to customize)")
    print("🔐 Requesting Turnkey to sign...")
    
    try:
        msg_resp = client.sign_message(sign_with=sign_with, message=msg, use_keccak256=True)
        msg_sign_ok = True
        print("✅ Message signed")
        print("📋 Signed Message (raw response):")
        print(json.dumps(msg_resp, indent=2))
    except Exception as e:
        print(f"❌ Message signing failed: {e}")
        print("💡 Fix tips:")
        print("   - Check policy restrictions on message signing")
        print("   - Try shorter ASCII text or keep use_keccak256=True (Ethereum convention)")
    
    print()

    # 3) EIP-712 Structured Data Signing  
    print("🧩 Step 3/4 · EIP-712 Structured Data Signing")
    print("-" * 40)
    typed_data = {
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
        "domain": {"name": "Turnkey", "version": "1", "chainId": 1},
        "message": {"contents": "hi"},
    }
    print("📊 EIP-712 structured data (example):")
    print(json.dumps(typed_data, indent=2))
    print("🔐 Requesting Turnkey to sign...")
    
    try:
        typed_resp = client.sign_typed_data(sign_with=sign_with, typed_data=typed_data)
        typed_ok = True
        print("✅ EIP-712 data signed")
        print("📋 Signed Typed Data (raw response):")
        print(json.dumps(typed_resp, indent=2))
    except Exception as e:
        print(f"❌ EIP-712 signing failed: {e}")
        print("💡 Fix tips:")
        print("   - Ensure typed_data schema matches EIP-712 and policy allows it")
        print("   - Keep numeric fields as strings to avoid type ambiguity")
    
    print()

    4) Activity Audit
    print("🧩 Step 4/4 · Activity Audit & History")
    print("-" * 40)
    print("📊 Querying recent signing activities...")
    
    try:
        acts = client.list_activities(limit="5")
        audit_ok = True
        print("✅ Activity history retrieved")
        print("📋 Recent Activities:")
        
        if acts.get("activities"):
            for i, activity in enumerate(acts["activities"][:3], 1):
                print(f"  {i}. Type: {activity.get('type', 'Unknown')}")
                print(f"     Status: {activity.get('status', 'Unknown')}")
                print(f"     Created: {activity.get('createdAt', 'Unknown')}")
                print()
        else:
            print("  No recent activities found.")
            
        print("📋 Full Response:")
        print(json.dumps(acts, indent=2))
    except Exception as e:
        print(f"❌ Activity query failed: {e}")
        print("💡 Fix tips:")
        print("   - Check network connectivity and Turnkey credentials")
        print("   - Ensure your keys have recent activities or increase limit")
    
    print()
    print("📊 Demo Summary")
    print("=" * 50)
    print(f"   • EVM tx signed:        {'✅' if tx_sign_ok else '❌'}")
    print(f"   • Broadcast attempted:  {'✅' if broadcast_ok else 'ℹ️  skipped'}")
    print(f"   • Message signed:       {'✅' if msg_sign_ok else '❌'}")
    print(f"   • EIP-712 signed:       {'✅' if typed_ok else '❌'}")
    print(f"   • Activity retrieved:   {'✅' if audit_ok else '❌'}")
    
    if not tx_sign_ok:
        print("   → Action: build a test tx and set TURNKEY_UNSIGNED_TX_HEX (see Step 1 quick start)")
    if tx_sign_ok and not broadcast_ok:
        print("   → Action: set WEB3_RPC_URL to auto-broadcast and verify on-chain")
    if not msg_sign_ok:
        print("   → Action: retry with a simple ASCII message or review policy")
    if not typed_ok:
        print("   → Action: validate typed data schema and policy permissions")
    if not audit_ok:
        print("   → Action: re-run after signing to populate recent activities")
    
    print("\n🎉 Demo complete. Your agent can now request secure signatures without exposing private keys.")


if __name__ == "__main__":
    main()
