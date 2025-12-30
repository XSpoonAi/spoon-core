"""
Environment security helpers for ENC:v2 encrypted secrets.

Security model:
- Decrypted secrets are stored in SecretVault, NOT in os.environ.
- The `decrypted_secrets` context manager provides scoped access to secrets
  via the vault, automatically wiping them when the context exits.
- Legacy `decrypted_environ` is DEPRECATED and will emit a warning.

This avoids the critical security issue where secrets remain in os.environ
(readable via /proc/{pid}/environ, inherited by subprocesses, etc.).
"""

from __future__ import annotations

import asyncio
import atexit
import getpass
import os
import sys
import warnings
import weakref
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, Iterable, Iterator, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

from spoon_ai.wallet.security import (
    ENCRYPTED_PREFIX_V2,
    DecryptionError,
    decrypt_and_store,
)
from spoon_ai.wallet.vault import SecretVault, get_vault

ENCRYPTED_PREFIX = ENCRYPTED_PREFIX_V2

_MASTER_PASSWORD_ENV = "SPOON_MASTER_PWD"
_DOTENV_LOADED = False
_DECRYPT_ON_IMPORT_ENV = "SPOON_SECURITY_DECRYPT_ON_IMPORT"

_LOOP_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_loop_lock() -> asyncio.Lock:
    """Return a per-event-loop lock to serialize vault operations."""
    loop = asyncio.get_running_loop()
    lock = _LOOP_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _LOOP_LOCKS[loop] = lock
    return lock


def _is_encrypted(value: str) -> bool:
    """Check if a value is an encrypted secret (ENC:v2 format)."""
    return value.startswith(ENCRYPTED_PREFIX_V2)


def _collect_encrypted_vars(env: Dict[str, str]) -> List[str]:
    """Find all environment variable names with encrypted values."""
    return [
        key
        for key, value in env.items()
        if isinstance(value, str) and _is_encrypted(value)
    ]


def _resolve_master_password(*, prompt: bool = True) -> str:
    """
    Resolve the master password from environment or interactive prompt.

    Note: Reading password from env var is less secure than interactive prompt,
    but may be necessary for CI/CD environments.
    """
    pwd = os.getenv(_MASTER_PASSWORD_ENV)
    if pwd:
        return pwd
    # Fallback: interactive prompt if TTY is available
    try:
        if prompt and sys.stdin.isatty():
            return getpass.getpass(
                f"Enter master password ({_MASTER_PASSWORD_ENV}) to decrypt secrets: "
            )
    except Exception:
        pass
    return ""


def _load_env_files() -> None:
    """Load .env files if python-dotenv is available."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED or load_dotenv is None:
        return

    candidates = [".env", "../.env", "../../.env"]
    for path in candidates:
        if os.path.exists(path):
            load_dotenv(path, override=False)
            _DOTENV_LOADED = True
            return
    # Fallback to default search
    load_dotenv()
    _DOTENV_LOADED = True


def _decrypt_to_vault(
    keys: Iterable[str],
    *,
    password: str,
    vault: SecretVault,
) -> List[str]:
    """
    Decrypt encrypted env vars and store in vault.

    Returns:
        List of vault keys that were successfully stored.
    """
    stored_keys: List[str] = []
    for key in keys:
        value = os.environ.get(key)
        if not (isinstance(value, str) and _is_encrypted(value)):
            continue
        try:
            decrypt_and_store(value, password, key, vault=vault)
            stored_keys.append(key)
        except DecryptionError:
            raise
    return stored_keys


def init_security(*, vault: Optional[SecretVault] = None) -> SecretVault:
    """
    Initialize env security features and return the vault.

    Behaviour:
    - Load `.env` files (if python-dotenv is installed).
    - Detect presence of ENC:v2 encrypted values.
    - If `SPOON_SECURITY_DECRYPT_ON_IMPORT` is truthy, decrypt all encrypted
      env vars and store in SecretVault (NOT os.environ).
    - Register atexit handler to wipe vault on shutdown.

    Returns:
        The SecretVault instance containing decrypted secrets.

    Raises:
        RuntimeError: If encrypted vars found but no password provided.
        DecryptionError: If decryption fails.
    """
    if vault is None:
        vault = get_vault()

    _load_env_files()
    encrypted_keys = _collect_encrypted_vars(os.environ)
    if not encrypted_keys:
        return vault

    # Mark that encrypted secrets were present (for wallet selection logic)
    os.environ["SPOON_ENC_PRESENT"] = "1"

    if not _truthy(os.getenv(_DECRYPT_ON_IMPORT_ENV)):
        return vault

    master_pwd = _resolve_master_password(prompt=True)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {encrypted_keys} but {_MASTER_PASSWORD_ENV} is not set. "
            f"Set {_MASTER_PASSWORD_ENV} or enable a TTY prompt to provide the password."
        )

    try:
        _decrypt_to_vault(encrypted_keys, password=master_pwd, vault=vault)
    except DecryptionError as exc:
        raise RuntimeError(f"Failed to decrypt environment secrets: {exc}") from exc

    # Register cleanup on exit
    atexit.register(vault.wipe_all)

    return vault


@contextmanager
def decrypted_secrets(
    keys: Optional[Iterable[str]] = None,
    *,
    password: Optional[str] = None,
    prompt: bool = True,
    vault: Optional[SecretVault] = None,
) -> Iterator[SecretVault]:
    """
    Context manager for scoped access to decrypted secrets via SecretVault.

    This is the RECOMMENDED way to access encrypted secrets. Secrets are:
    1. Decrypted and stored in the vault on context entry.
    2. Accessible via `vault.get_decoded(key)` within the context.
    3. Automatically wiped from the vault on context exit.

    Args:
        keys: Specific env var names to decrypt. If None, all encrypted vars.
        password: Master password. If None, resolved from env or prompt.
        prompt: Whether to prompt for password if not in env.
        vault: Optional vault instance (defaults to singleton).

    Yields:
        SecretVault instance with decrypted secrets.

    Example:
        with decrypted_secrets(["PRIVATE_KEY"]) as vault:
            with vault.get_decoded("PRIVATE_KEY") as pk:
                account = Account.from_key(pk)
                # Use account...
        # Secrets wiped automatically
    """
    if vault is None:
        vault = get_vault()

    _load_env_files()
    env = os.environ
    target_keys = list(keys) if keys is not None else _collect_encrypted_vars(env)
    target_keys = [
        k for k in target_keys if isinstance(env.get(k), str) and _is_encrypted(env[k])
    ]

    if not target_keys:
        yield vault
        return

    master_pwd = password or _resolve_master_password(prompt=prompt)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {target_keys} but {_MASTER_PASSWORD_ENV} is not set."
        )

    stored_keys: List[str] = []
    try:
        stored_keys = _decrypt_to_vault(target_keys, password=master_pwd, vault=vault)
        yield vault
    finally:
        # Wipe only the keys we stored in this context
        for k in stored_keys:
            vault.wipe(k)


@asynccontextmanager
async def async_decrypted_secrets(
    keys: Optional[Iterable[str]] = None,
    *,
    password: Optional[str] = None,
    prompt: bool = True,
    vault: Optional[SecretVault] = None,
) -> Iterator[SecretVault]:
    """
    Async version of `decrypted_secrets()` with per-event-loop locking.

    This prevents race conditions when multiple async tasks attempt to
    decrypt/wipe secrets concurrently.

    Args:
        keys: Specific env var names to decrypt. If None, all encrypted vars.
        password: Master password. If None, resolved from env or prompt.
        prompt: Whether to prompt for password if not in env.
        vault: Optional vault instance (defaults to singleton).

    Yields:
        SecretVault instance with decrypted secrets.
    """
    if vault is None:
        vault = get_vault()

    _load_env_files()
    env = os.environ
    target_keys = list(keys) if keys is not None else _collect_encrypted_vars(env)
    target_keys = [
        k for k in target_keys if isinstance(env.get(k), str) and _is_encrypted(env[k])
    ]

    if not target_keys:
        yield vault
        return

    master_pwd = password or _resolve_master_password(prompt=prompt)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {target_keys} but {_MASTER_PASSWORD_ENV} is not set."
        )

    lock = _get_loop_lock()
    stored_keys: List[str] = []

    async with lock:
        try:
            stored_keys = _decrypt_to_vault(
                target_keys, password=master_pwd, vault=vault
            )
            yield vault
        finally:
            for k in stored_keys:
                vault.wipe(k)


# =============================================================================
# DEPRECATED: Legacy functions that modify os.environ
# =============================================================================


@contextmanager
def decrypted_environ(
    keys: Optional[Iterable[str]] = None,
    *,
    password: Optional[str] = None,
    prompt: bool = True,
) -> Iterator[None]:
    """
    DEPRECATED: Use `decrypted_secrets()` instead.

    This function modifies os.environ, which is a security risk.
    Secrets in os.environ can be read via /proc/{pid}/environ.
    """
    warnings.warn(
        "decrypted_environ() is deprecated due to security concerns. "
        "Use decrypted_secrets() which stores secrets in SecretVault instead of os.environ.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import here to avoid circular import
    from spoon_ai.wallet.security import decrypt_private_key

    _load_env_files()
    env = os.environ
    target_keys = list(keys) if keys is not None else _collect_encrypted_vars(env)
    target_keys = [
        k for k in target_keys if isinstance(env.get(k), str) and _is_encrypted(env[k])
    ]

    if not target_keys:
        yield
        return

    master_pwd = password or _resolve_master_password(prompt=prompt)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {target_keys} but {_MASTER_PASSWORD_ENV} is not set."
        )

    original = {k: env[k] for k in target_keys}
    try:
        for key in target_keys:
            value = env.get(key)
            if value and _is_encrypted(value):
                env[key] = decrypt_private_key(value, master_pwd)
        yield
    finally:
        for k, v in original.items():
            env[k] = v


@asynccontextmanager
async def async_decrypted_environ(
    keys: Optional[Iterable[str]] = None,
    *,
    password: Optional[str] = None,
    prompt: bool = True,
) -> Iterator[None]:
    """
    DEPRECATED: Use `async_decrypted_secrets()` instead.

    This function modifies os.environ, which is a security risk.
    """
    warnings.warn(
        "async_decrypted_environ() is deprecated due to security concerns. "
        "Use async_decrypted_secrets() which stores secrets in SecretVault instead of os.environ.",
        DeprecationWarning,
        stacklevel=2,
    )

    from spoon_ai.wallet.security import decrypt_private_key

    _load_env_files()
    env = os.environ
    target_keys = list(keys) if keys is not None else _collect_encrypted_vars(env)
    target_keys = [
        k for k in target_keys if isinstance(env.get(k), str) and _is_encrypted(env[k])
    ]

    if not target_keys:
        yield
        return

    master_pwd = password or _resolve_master_password(prompt=prompt)
    if not master_pwd:
        raise RuntimeError(
            f"Found encrypted environment variables {target_keys} but {_MASTER_PASSWORD_ENV} is not set."
        )

    lock = _get_loop_lock()
    async with lock:
        original = {k: env[k] for k in target_keys}
        try:
            for key in target_keys:
                value = env.get(key)
                if value and _is_encrypted(value):
                    env[key] = decrypt_private_key(value, master_pwd)
            yield
        finally:
            for k, v in original.items():
                env[k] = v
