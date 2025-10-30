"""
Demonstration of SpoonOS agents interacting with the x402 payment service.

This walkthrough mirrors the ChaosChain Genesis Studio x402 flow:
1. Load configuration (config.json + .env overrides).
2. Build a 0.01 USDC payment request for https://www.x402.org/protected on Base Sepolia.
3. Sign the X-PAYMENT header using the configured agent wallet.
4. Inspect the simulated 402 challenge and settlement receipt.
5. Show how agent tools automate the retry flow (without calling the live facilitator).

Run with:
    uv run python examples/x402_agent_demo.py
"""

import asyncio
import json
from typing import Any, Dict

from x402.encoding import safe_base64_encode

from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.payments import X402PaymentRequest, X402PaymentService
from spoon_ai.tools.x402_payment import X402PaywalledRequestTool


def _print_section(title: str, payload: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


async def demonstrate_paywall_handshake():
    service = X402PaymentService()

    payment_request = X402PaymentRequest(
        amount_usdc=0.01,
        resource="https://www.x402.org/protected",
        description="Protected demo content",
        memo="Base Sepolia demo payment",
        currency="USDC",
        metadata={
            "network": "base-sepolia",
            "docs": "https://docs.proxy402.com/",
            "faucet": "https://faucet.circle.com/",
        },
        output_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}, "timestamp": {"type": "string"}},
        },
    )

    requirements = service.build_payment_requirements(payment_request)
    header = service.build_payment_header(requirements, max_value=int(requirements.max_amount_required))
    simulated_402 = service.build_payment_required_response(
        "Payment required for this SpoonOS action.", request=payment_request
    ).model_dump(by_alias=True)

    _print_section("x402 Payment Requirements", requirements.model_dump(by_alias=True))
    print("\nSigned X-PAYMENT header:\n", header)
    _print_section("Simulated 402 Challenge", simulated_402)

    simulated_receipt_raw = {
        "success": True,
        "transaction": "0x402demo00000000000000000000000000000000000000000000000000000000001",
        "network": requirements.network,
        "payer": "0xYourAgentAddress",
    }
    encoded_receipt = safe_base64_encode(json.dumps(simulated_receipt_raw))
    receipt = service.decode_payment_response(encoded_receipt).model_dump(exclude_none=True)
    _print_section("Decoded X-PAYMENT-RESPONSE Receipt", receipt)


async def demonstrate_tool_usage():
    agent = SpoonReactAI(name="x402-demo-agent")
    await agent.initialize()

    tool = agent.avaliable_tools.get_tool("x402_paywalled_request")
    if not isinstance(tool, X402PaywalledRequestTool):
        print("x402 tool not available; check configuration (X402_AGENT_PRIVATE_KEY, etc.).")
        return

    create_tool = agent.avaliable_tools.get_tool("x402_create_payment")
    if create_tool is None:
        print("x402_create_payment tool not available; ensure x402 configuration is loaded.")
        return
    base_requirements = create_tool.service.build_payment_requirements()  # type: ignore[attr-defined]
    paywall_payload = create_tool.service.build_payment_required_response(  # type: ignore[attr-defined]
        "USDC payment required to continue."
    ).model_dump(by_alias=True)

    _print_section("Mock 402 Paywall Payload", paywall_payload)
    print(
        "\nUse `x402_paywalled_request` to retry automatically:\n"
        "  await tool.execute(\n"
        "      url='https://www.x402.org/protected',\n"
        "      amount_usdc=0.01,\n"
        "      memo='Base Sepolia demo payment',\n"
        "      metadata={'demo': 'chaoschain-aligned'},\n"
        "  )\n"
    )
    _print_section("Default Requirements (for reference)", base_requirements.model_dump(by_alias=True))


async def main():
    await demonstrate_paywall_handshake()
    await demonstrate_tool_usage()


if __name__ == "__main__":
    asyncio.run(main())
