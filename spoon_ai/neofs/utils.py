import base64
import binascii
import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from neo3.core import cryptography
from neo3.wallet.account import Account


PREFIX = bytes.fromhex("010001f0")
SUFFIX = bytes.fromhex("0000")


class SignatureError(Exception):
    """Raised when signature payload construction fails."""


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("VarInt cannot encode negative values")
    if value < 0xFD:
        return value.to_bytes(1, "little")
    if value <= 0xFFFF:
        return b"\xFD" + value.to_bytes(2, "little")
    if value <= 0xFFFFFFFF:
        return b"\xFE" + value.to_bytes(4, "little")
    return b"\xFF" + value.to_bytes(8, "little")


def _build_serialized_message(parameters: Iterable[bytes]) -> bytes:
    concatenated = b"".join(parameters)
    length_prefix = _encode_varint(len(concatenated))
    return PREFIX + length_prefix + concatenated + SUFFIX


@dataclass
class SignatureComponents:
    signature: str
    salt: str
    public_key: str

    def signature_header(self) -> str:
        return f"{self.signature}{self.salt}"


def sign_with_salt(private_key_wif: str, *payload_parts: bytes, salt: bytes | None = None) -> SignatureComponents:
    account = Account.from_wif(private_key_wif)
    salt_bytes = salt if salt is not None else os.urandom(16)
    serialized_message = _build_serialized_message((salt_bytes, *payload_parts))
    signature = account.sign(serialized_message)
    return SignatureComponents(
        signature=signature.hex(),
        salt=salt_bytes.hex(),
        public_key=account.public_key.to_array().hex(),
    )


def generate_simple_signature_params(private_key_wif: Optional[str] = None, payload_parts: Iterable[bytes] | None = None, *, components: SignatureComponents | None = None, salt: bytes | None = None) -> dict:
    if components is None:
        if private_key_wif is None:
            raise ValueError("Either components or private_key_wif must be provided")
        parts = tuple(payload_parts or ())
        components = sign_with_salt(private_key_wif, *parts, salt=salt)
    return {
        "signatureParam": components.signature_header(),
        "signatureKeyParam": components.public_key,
        "signatureScheme": "ECDSA_SHA256",
    }


# utils.py (or wherever you place sign_bearer_token)
import base64, binascii, os
from neo3.core import cryptography  # Use neo3's sign
from cryptography.hazmat.primitives import hashes

# WalletConnect fixed prefix and postfix
_WC_PREFIX = b"\x01\x00\x01\xf0"
_WC_POSTFIX = b"\x00\x00"


import base64, binascii

def _wc_build_message(token_b64: str, salt: bytes) -> tuple[bytes, bytes]:
    prefix  = b"\x01\x00\x01\xf0"
    postfix = b"\x00\x00"
    token_raw  = base64.standard_b64decode(token_b64)
    normalized = base64.standard_b64encode(token_raw)  
    hex_salt   = binascii.hexlify(salt)                

    msg_len = len(hex_salt) + len(normalized)
    if msg_len >= 256:
        raise ValueError(f"WalletConnect message too long: {msg_len}")
    msg = prefix + msg_len.to_bytes(1, "big") + hex_salt + normalized + postfix
    return msg, hex_salt


# utils.py
import os, base64, binascii, hashlib
from neo3.wallet.account import Account
from neo3.core import cryptography

def sign_bearer_token(bearer_token: str, private_key_wif: str, *, wallet_connect: bool = True) -> tuple[str, str]:
    """
    Returns (signature_hex, compressed_pubkey_hex)

    - wallet_connect=True:
        msg = WC format (with prefix/len/salt/postfix), hash=SHA-256
        X-Bearer-Signature = <DER signature hex> + <16B salt hex>
        X-Bearer-Signature-Key = <compressed public key hex>
        URL needs to append ?walletConnect=true
    """
    acct = Account.from_wif(private_key_wif)
    pubkey_hex = acct.public_key.to_array().hex()  # 33B compressed public key → hex

    if wallet_connect:
        salt = os.urandom(16)
        msg, hex_salt = _wc_build_message(bearer_token, salt)
        der_sig = cryptography.sign(msg, acct.private_key, hash_func=hashlib.sha256)
        sig_hex = binascii.hexlify(der_sig).decode() + hex_salt.decode()
        return sig_hex, pubkey_hex


    token_raw = base64.standard_b64decode(bearer_token)
    der_sig = cryptography.sign(token_raw, acct.private_key, hash_func=hashlib.sha512)
    sig_hex = "04" + binascii.hexlify(der_sig).decode()
    return sig_hex, pubkey_hex
