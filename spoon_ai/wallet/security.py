"""
AES-GCM helpers for encrypting and decrypting private keys.

Security Features:
- AES-256-GCM authenticated encryption
- Argon2id key derivation (memory-hard, resistant to GPU/ASIC attacks)
- Random salt and IV per encryption
- Secure memory handling with bytearray zero-fill
- No plaintext returned; secrets stored directly in SecretVault

Dependencies:
- cryptography >= 41.0.0 (for Argon2id support)

Encrypted format: ENC:v2:<base64(salt || iv || ciphertext || tag)>
"""

from __future__ import annotations

import base64
import os
from typing import Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Argon2 import with fallback check
try:
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

    _ARGON2_AVAILABLE = True
except ImportError:
    _ARGON2_AVAILABLE = False
    Argon2id = None  # type: ignore

from .vault import SecretVault


# Version prefix
ENCRYPTED_PREFIX_V2 = "ENC:v2:"  # Argon2id
ENCRYPTED_PREFIX = ENCRYPTED_PREFIX_V2  # Default for encryptions

# Cryptographic parameters
_SALT_BYTES = 16
_IV_BYTES = 12  # 96 bits for GCM
_KEY_BYTES = 32  # AES-256

# Argon2id parameters
# OWASP recommendations: t=3, m=65536 (64 MiB), p=4
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65536  # 64 MiB
_ARGON2_PARALLELISM = 4


class DecryptionError(ValueError):
    """
    Raised when decryption fails.

    This is a generic error that does not reveal whether the failure was due to
    an incorrect password, corrupted data, or authentication tag mismatch.
    """

    pass


def _zero_fill(buf: bytearray) -> None:
    """Zero-fill a bytearray in-place."""
    for i in range(len(buf)):
        buf[i] = 0


def _derive_key_argon2(password: str, salt: bytes) -> bytearray:
    """
    Derive a symmetric key using Argon2id.

    Argon2id is the recommended KDF for password hashing as of 2024.
    It combines Argon2i (resistant to side-channel attacks) and Argon2d
    (resistant to GPU cracking).

    Returns:
        bytearray: Derived key (caller must zero-fill after use).

    Raises:
        RuntimeError: If Argon2id is not available in the cryptography library.
    """
    if not _ARGON2_AVAILABLE:
        raise RuntimeError(
            "Argon2id requires cryptography >= 41.0.0. "
            "Please upgrade: pip install 'cryptography>=41.0.0'"
        )

    if not password:
        raise ValueError("Password is required to derive encryption key.")

    kdf = Argon2id(
        salt=salt,
        length=_KEY_BYTES,
        iterations=_ARGON2_TIME_COST,
        lanes=_ARGON2_PARALLELISM,
        memory_cost=_ARGON2_MEMORY_COST,
    )
    # Return as bytearray for secure wiping
    return bytearray(kdf.derive(password.encode("utf-8")))


def _split_payload(payload: bytes) -> Tuple[bytes, bytes, bytes]:
    """Split encrypted payload into salt, IV, and ciphertext."""
    min_length = _SALT_BYTES + _IV_BYTES + 16  # Minimum: salt + iv + 1 block
    if len(payload) < min_length:
        raise ValueError("Encrypted payload is malformed or truncated.")

    salt = payload[:_SALT_BYTES]
    iv = payload[_SALT_BYTES : _SALT_BYTES + _IV_BYTES]
    ciphertext = payload[_SALT_BYTES + _IV_BYTES :]
    return salt, iv, ciphertext


def encrypt_private_key(raw_private_key: str, password: str) -> str:
    """
    Encrypt a private key using AES-256-GCM with Argon2id key derivation.

    Args:
        raw_private_key: The plaintext private key to encrypt.
        password: Password used to derive the encryption key.

    Returns:
        Encrypted payload: ENC:v2:<base64(...)>

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If Argon2id is not available.
    """
    if not raw_private_key:
        raise ValueError("raw_private_key is required for encryption.")
    if not password:
        raise ValueError("password is required for encryption.")

    salt = os.urandom(_SALT_BYTES)
    iv = os.urandom(_IV_BYTES)

    key = _derive_key_argon2(password, salt)

    try:
        aesgcm = AESGCM(bytes(key))
        ciphertext = aesgcm.encrypt(iv, raw_private_key.encode("utf-8"), None)
        payload = base64.urlsafe_b64encode(salt + iv + ciphertext).decode("utf-8")
        return f"{ENCRYPTED_PREFIX_V2}{payload}"
    finally:
        # Securely wipe the derived key
        _zero_fill(key)


def decrypt_private_key(enc_value: str, password: str) -> str:
    """
    Decrypt a value produced by `encrypt_private_key`.

    WARNING: This function returns a Python str, which cannot be securely wiped.
    Prefer `decrypt_and_store()` for production use.

    Args:
        enc_value: Encrypted value with ENC:v2 prefix.
        password: Password used to derive the key.

    Returns:
        Decrypted private key as a string.

    Raises:
        DecryptionError: If decryption fails (wrong password or corrupted data).
    """
    key: Optional[bytearray] = None

    try:
        if not enc_value.startswith(ENCRYPTED_PREFIX_V2):
            raise DecryptionError("Invalid encrypted format. Expected ENC:v2 prefix.")

        payload_b64 = enc_value[len(ENCRYPTED_PREFIX_V2) :]

        if not password:
            raise DecryptionError("Invalid password or corrupted data.")

        payload = base64.urlsafe_b64decode(payload_b64)
        salt, iv, ciphertext = _split_payload(payload)

        key = _derive_key_argon2(password, salt)

        aesgcm = AESGCM(bytes(key))
        plaintext = aesgcm.decrypt(iv, ciphertext, None)
        return plaintext.decode("utf-8")

    except DecryptionError:
        raise
    except Exception:
        # Catch all crypto exceptions and convert to generic error
        # This prevents oracle attacks by not revealing failure reason
        raise DecryptionError("Invalid password or corrupted data.") from None
    finally:
        if key is not None:
            _zero_fill(key)


def decrypt_and_store(
    enc_value: str,
    password: str,
    vault_key: str,
    *,
    vault: Optional[SecretVault] = None,
) -> None:
    """
    Decrypt a private key and store it directly in the SecretVault.

    This is the RECOMMENDED way to decrypt secrets. Unlike `decrypt_private_key()`,
    this function:
    1. Never returns the plaintext (no risk of accidental exposure).
    2. Stores the secret as a mutable bytearray that can be wiped later.
    3. Zeros all intermediate key material before returning.

    Args:
        enc_value: Encrypted value with ENC:v2 prefix.
        password: Password used to derive the key.
        vault_key: Key under which to store the decrypted secret in the vault.
        vault: Optional SecretVault instance (defaults to singleton).

    Raises:
        DecryptionError: If decryption fails (wrong password or corrupted data).

    Example:
        from spoon_ai.wallet.vault import SecretVault
        from spoon_ai.wallet.security import decrypt_and_store

        vault = SecretVault()
        decrypt_and_store(encrypted_key, password, "eth_private_key", vault=vault)

        # Later, when needed:
        with vault.get_decoded("eth_private_key") as pk:
            sign_transaction(pk)

        # On shutdown:
        vault.wipe_all()
    """
    if vault is None:
        vault = SecretVault()

    key: Optional[bytearray] = None
    plaintext_buf: Optional[bytearray] = None

    try:
        if not enc_value.startswith(ENCRYPTED_PREFIX_V2):
            raise DecryptionError("Invalid encrypted format. Expected ENC:v2 prefix.")

        payload_b64 = enc_value[len(ENCRYPTED_PREFIX_V2) :]

        if not password:
            raise DecryptionError("Invalid password or corrupted data.")

        payload = base64.urlsafe_b64decode(payload_b64)
        salt, iv, ciphertext = _split_payload(payload)

        key = _derive_key_argon2(password, salt)

        aesgcm = AESGCM(bytes(key))
        plaintext_bytes = aesgcm.decrypt(iv, ciphertext, None)

        # Convert to bytearray for secure storage
        plaintext_buf = bytearray(plaintext_bytes)

        # Store in vault (vault.store() makes its own copy)
        vault.store(vault_key, plaintext_buf)

    except DecryptionError:
        raise
    except Exception:
        raise DecryptionError("Invalid password or corrupted data.") from None
    finally:
        # Securely wipe all intermediate buffers
        if key is not None:
            _zero_fill(key)
        if plaintext_buf is not None:
            _zero_fill(plaintext_buf)
