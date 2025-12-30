#!/usr/bin/env python
"""
Interactive helper to produce ENC:v2 payloads for PRIVATE_KEY (or any secret).

Uses AES-256-GCM encryption with Argon2id key derivation.
"""

import getpass
import sys

from spoon_ai.wallet.security import encrypt_private_key


def main() -> int:
    batch = input("Batch mode? (y/N): ").strip().lower().startswith("y")

    if batch:
        raw_values = input("Enter secrets separated by space: ").strip().split()
        if not raw_values:
            print("At least one secret is required.", file=sys.stderr)
            return 1
    else:
        raw_value = getpass.getpass("Plaintext private key (0x...): ").strip()
        if not raw_value:
            print("A private key is required.", file=sys.stderr)
            return 1
        raw_values = [raw_value]

    password = getpass.getpass("Password: ")
    if not password:
        print("A password is required.", file=sys.stderr)
        return 1

    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords do not match.", file=sys.stderr)
        return 1

    print("\nEncrypted values (order matches input):")
    for idx, raw_key in enumerate(raw_values, start=1):
        encrypted = encrypt_private_key(raw_key, password)
        label = f"SECRET_{idx}" if batch else "PRIVATE_KEY"
        print(f"{label}={encrypted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
