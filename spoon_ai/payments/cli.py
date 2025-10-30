from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict

from spoon_ai.payments import (
    X402PaymentRequest,
    X402PaymentService,
)


def _parse_json_object(value: str) -> Dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return data


def _build_request_from_args(args: argparse.Namespace) -> X402PaymentRequest:
    payload: Dict[str, Any] = {}
    for field in ("resource", "description", "mime_type", "scheme", "network", "pay_to", "timeout_seconds"):
        value = getattr(args, field, None)
        if value is not None:
            payload[field] = value
    if args.amount_usdc is not None:
        payload["amount_usdc"] = args.amount_usdc
    if args.amount_atomic is not None:
        payload["amount_atomic"] = args.amount_atomic
    for field in ("currency", "memo", "payer"):
        value = getattr(args, field, None)
        if value is not None:
            payload[field] = value
    if args.metadata is not None:
        payload["metadata"] = args.metadata
    if args.output_schema is not None:
        payload["output_schema"] = args.output_schema
    return X402PaymentRequest(**payload)


async def handle_requirements(args: argparse.Namespace) -> None:
    service = X402PaymentService()
    request = _build_request_from_args(args)
    requirements = service.build_payment_requirements(request)
    print(json.dumps(requirements.model_dump(by_alias=True, exclude_none=True), indent=2))


async def handle_sign(args: argparse.Namespace) -> None:
    service = X402PaymentService()
    request = _build_request_from_args(args)
    requirements = service.build_payment_requirements(request)
    header = service.build_payment_header(requirements, max_value=args.max_value)
    output = {
        "header": header,
        "requirements": requirements.model_dump(by_alias=True, exclude_none=True),
    }
    print(json.dumps(output, indent=2))


async def handle_verify(args: argparse.Namespace) -> None:
    service = X402PaymentService()
    requirements = service.build_payment_requirements()
    result = await service.verify_payment(args.header, requirements)
    print(json.dumps(result.model_dump(exclude_none=True), indent=2))


async def handle_settle(args: argparse.Namespace) -> None:
    service = X402PaymentService()
    requirements = service.build_payment_requirements()
    outcome = await service.verify_and_settle(args.header, requirements, settle=not args.skip_settle)
    print(json.dumps(outcome.model_dump(exclude_none=True), indent=2))


COMMAND_HANDLERS = {
    "requirements": handle_requirements,
    "sign": handle_sign,
    "verify": handle_verify,
    "settle": handle_settle,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpoonOS x402 utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(sub):
        sub.add_argument("--resource", help="Resource URL")
        sub.add_argument("--description", help="Payment description")
        sub.add_argument("--mime-type", dest="mime_type", help="Resource MIME type")
        sub.add_argument("--amount-usdc", dest="amount_usdc", type=float, help="Amount to charge in USD")
        sub.add_argument("--amount-atomic", dest="amount_atomic", type=int, help="Amount override in atomic units")
        sub.add_argument("--scheme", help="Payment scheme")
        sub.add_argument("--network", help="Target network")
        sub.add_argument("--pay-to", dest="pay_to", help="Recipient wallet")
        sub.add_argument("--timeout-seconds", dest="timeout_seconds", type=int, help="Validity window for the payment")
        sub.add_argument("--currency", help="Currency label advertised to the payer")
        sub.add_argument("--memo", help="Memo or purpose string recorded with the payment")
        sub.add_argument("--payer", help="Payer identifier or account label")
        sub.add_argument("--metadata", type=_parse_json_object, help="Additional metadata merged into the x402 extra payload (JSON object)")
        sub.add_argument("--output-schema", dest="output_schema", type=_parse_json_object, help="JSON schema advertised for the paywalled response")

    req_parser = subparsers.add_parser("requirements", help="Generate payment requirements JSON")
    add_common_arguments(req_parser)

    sign_parser = subparsers.add_parser("sign", help="Generate signed X-PAYMENT header")
    add_common_arguments(sign_parser)
    sign_parser.add_argument("--max-value", dest="max_value", type=int, help="Safety ceiling for signed payments")

    verify_parser = subparsers.add_parser("verify", help="Verify a provided X-PAYMENT header")
    verify_parser.add_argument("header", help="Base64-encoded X-PAYMENT header value")

    settle_parser = subparsers.add_parser("settle", help="Verify (and optionally settle) a payment header")
    settle_parser.add_argument("header", help="Base64-encoded X-PAYMENT header value")
    settle_parser.add_argument("--skip-settle", action="store_true", help="Only verify, do not settle")

    return parser


async def main_async(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMAND_HANDLERS[args.command]
    await handler(args)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(main_async(argv))


if __name__ == "__main__":
    main()
