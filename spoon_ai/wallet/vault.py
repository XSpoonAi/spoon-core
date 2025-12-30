"""
Thread-safe in-memory vault for sensitive secrets.

Security model:
- Secrets are stored as mutable `bytearray` objects, allowing explicit memory wiping.
- No secret is ever written to os.environ or any persistent storage.
- The `get_decoded` context manager provides temporary access to decoded strings
  with automatic cleanup of references (though Python str objects cannot be
  forcefully zeroed due to immutability).
- All operations are thread-safe via a reentrant lock.

Usage:
    vault = SecretVault()
    vault.store("my_key", b"secret_data")

    with vault.get_decoded("my_key") as secret_str:
        # Use secret_str briefly
        do_something(secret_str)
    # Reference released after context exit

    vault.wipe("my_key")  # Securely wipe when done
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Union


class SecretVault:
    """
    Thread-safe singleton vault for storing sensitive secrets in memory.

    Secrets are stored as `bytearray` objects, which can be explicitly zeroed
    after use to minimize the window of exposure in memory.

    Thread Safety:
        All public methods acquire a reentrant lock, making this class safe
        for concurrent access from multiple threads.

    Memory Security:
        - Uses `bytearray` instead of `bytes` to allow in-place zeroing.
        - `wipe()` and `wipe_all()` zero-fill before deletion.
        - `get_decoded()` context manager limits the lifetime of decoded strings.

    Limitations:
        - Python's garbage collector may not immediately reclaim memory.
        - Decoded `str` objects returned by `get_decoded()` are immutable and
          cannot be forcefully wiped; use them briefly and avoid storing references.
        - Memory may still be swapped to disk; consider using `mlock()` at OS level
          for high-security deployments.
    """

    _instance: Optional[SecretVault] = None
    _creation_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> SecretVault:
        """Ensure singleton pattern with thread-safe lazy initialization."""
        if cls._instance is None:
            with cls._creation_lock:
                # Double-checked locking
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._secrets: Dict[str, bytearray] = {}
                    instance._lock: threading.RLock = threading.RLock()
                    cls._instance = instance
        return cls._instance

    def store(self, key: str, value: Union[bytes, bytearray, str]) -> None:
        """
        Store a secret in the vault.

        Args:
            key: Identifier for the secret.
            value: Secret data as bytes, bytearray, or str.
                   Strings are encoded as UTF-8.

        Note:
            If a secret already exists under `key`, it will be wiped before
            being replaced.
        """
        if not key:
            raise ValueError("Secret key cannot be empty.")

        # Convert to bytearray for mutable storage
        if isinstance(value, str):
            data = bytearray(value.encode("utf-8"))
        elif isinstance(value, bytes):
            data = bytearray(value)
        elif isinstance(value, bytearray):
            data = bytearray(value)  # Copy to avoid external mutation
        else:
            raise TypeError(f"Value must be bytes, bytearray, or str; got {type(value).__name__}")

        with self._lock:
            # Wipe existing secret if present
            if key in self._secrets:
                self._zero_fill(self._secrets[key])
            self._secrets[key] = data

    def get_raw(self, key: str) -> Optional[bytes]:
        """
        Retrieve a secret as immutable bytes.

        Args:
            key: Identifier for the secret.

        Returns:
            Copy of the secret as bytes, or None if not found.

        Warning:
            The returned `bytes` object cannot be wiped. Prefer `get_decoded()`
            context manager for controlled access.
        """
        with self._lock:
            buf = self._secrets.get(key)
            return bytes(buf) if buf is not None else None

    @contextmanager
    def get_decoded(self, key: str, encoding: str = "utf-8") -> Iterator[Optional[str]]:
        """
        Context manager for temporary access to a decoded secret string.

        This is the preferred way to access secrets when a string is needed.
        The context manager ensures that:
        1. The secret is decoded only within the `with` block.
        2. Local references are explicitly deleted after the block exits.

        Args:
            key: Identifier for the secret.
            encoding: Character encoding (default: UTF-8).

        Yields:
            Decoded string, or None if the secret doesn't exist.

        Example:
            with vault.get_decoded("private_key") as pk:
                if pk:
                    sign_transaction(pk)
            # `pk` reference is now invalid

        Warning:
            Python strings are immutable and cannot be forcefully wiped from memory.
            Keep the usage window as short as possible and avoid copying the string
            to other variables or data structures.
        """
        decoded: Optional[str] = None
        try:
            with self._lock:
                buf = self._secrets.get(key)
                if buf is not None:
                    decoded = buf.decode(encoding)
            yield decoded
        finally:
            # Explicitly delete local reference (does not wipe memory, but
            # removes one reference to allow GC to potentially collect sooner)
            del decoded

    def exists(self, key: str) -> bool:
        """Check if a secret exists in the vault."""
        with self._lock:
            return key in self._secrets

    def keys(self) -> list[str]:
        """Return a list of all secret keys (not the values)."""
        with self._lock:
            return list(self._secrets.keys())

    def wipe(self, key: str) -> bool:
        """
        Securely wipe a secret from the vault.

        This method:
        1. Zero-fills the bytearray in-place.
        2. Removes the key from the vault.

        Args:
            key: Identifier for the secret to wipe.

        Returns:
            True if the secret existed and was wiped, False otherwise.
        """
        with self._lock:
            buf = self._secrets.pop(key, None)
            if buf is not None:
                self._zero_fill(buf)
                return True
            return False

    def wipe_all(self) -> int:
        """
        Securely wipe all secrets from the vault.

        Returns:
            Number of secrets that were wiped.
        """
        with self._lock:
            count = len(self._secrets)
            for buf in self._secrets.values():
                self._zero_fill(buf)
            self._secrets.clear()
            return count

    @staticmethod
    def _zero_fill(buf: bytearray) -> None:
        """
        Zero-fill a bytearray in-place.

        This is a best-effort security measure. Note that:
        - The Python runtime may have made copies during operations.
        - Memory may have been swapped to disk before this call.
        - A sophisticated attacker with memory access could potentially
          recover data through forensic analysis.
        """
        for i in range(len(buf)):
            buf[i] = 0

    def __repr__(self) -> str:
        with self._lock:
            return f"<SecretVault keys={list(self._secrets.keys())}>"

    def __contains__(self, key: str) -> bool:
        return self.exists(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._secrets)


# Module-level convenience function
def get_vault() -> SecretVault:
    """Get the singleton SecretVault instance."""
    return SecretVault()
