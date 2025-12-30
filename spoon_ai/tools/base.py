from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field


# Module-level flag to track if secrets have been decrypted this session
_SECRETS_INITIALIZED: bool = False


class BaseTool(ABC, BaseModel):
    name: str = Field(description="The name of the tool")
    description: str = Field(description="A description of the tool")
    parameters: dict = Field(description="The parameters of the tool")

    # When True, tool calls will run inside a scoped env-decryption context so
    # ENC:v2 secrets can be consumed by tools. Secrets are stored in SecretVault
    # (not os.environ) for better security.
    requires_decrypted_env: ClassVar[bool] = False

    # Heuristic default: common blockchain tool prefixes that may need private keys.
    _DECRYPT_ENV_NAME_PREFIXES: ClassVar[tuple[str, ...]] = (
        "evm_",
        "solana_",
        "neo",
        "neofs",
        "sign_",
        "broadcast_",
    )

    model_config = {
        "arbitrary_types_allowed": True
    }

    async def __call__(self, *args, **kwargs) -> Any:
        # Avoid decrypting secrets for unrelated tools (which would prompt for a
        # password unnecessarily). Use either an explicit opt-in flag on the tool
        # class or a conservative name-based heuristic.
        tool_name = (self.name or "").lower()
        should_decrypt = (
            self.requires_decrypted_env
            or tool_name.startswith(self._DECRYPT_ENV_NAME_PREFIXES)
            or "neofs" in tool_name
        )
        if not should_decrypt:
            return await self.execute(*args, **kwargs)

        # Ensure secrets are decrypted and available in vault (only once per session)
        await _ensure_secrets_initialized()
        return await self.execute(*args, **kwargs)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def to_param(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


async def _ensure_secrets_initialized() -> None:
    """
    Ensure encrypted environment secrets are decrypted and stored in SecretVault.

    This function is called once per session. Subsequent calls are no-ops.
    Secrets are stored in SecretVault (not os.environ) for security.
    """
    global _SECRETS_INITIALIZED

    if _SECRETS_INITIALIZED:
        return

    import os
    from spoon_ai.wallet.vault import get_vault
    from spoon_ai.wallet.security import (
        ENCRYPTED_PREFIX_V2,
        decrypt_and_store,
        DecryptionError,
    )

    vault = get_vault()

    # Find all encrypted env vars (ENC:v2 format)
    encrypted_keys = [
        key for key, value in os.environ.items()
        if isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX_V2)
    ]

    if not encrypted_keys:
        _SECRETS_INITIALIZED = True
        return

    # Resolve master password (prompt once)
    import getpass
    import sys

    password = os.getenv("SPOON_MASTER_PWD")
    if not password:
        try:
            if sys.stdin.isatty():
                password = getpass.getpass(
                    f"Enter master password to decrypt {len(encrypted_keys)} secret(s): "
                )
        except Exception:
            pass

    if not password:
        raise RuntimeError(
            f"Found encrypted secrets {encrypted_keys} but no password provided. "
            "Set SPOON_MASTER_PWD or run in a TTY to enter password interactively."
        )

    # Decrypt all secrets and store in vault
    for key in encrypted_keys:
        enc_value = os.environ.get(key)
        if not enc_value:
            continue

        # Check if already in vault (avoid re-decryption)
        if vault.exists(key):
            continue

        try:
            decrypt_and_store(enc_value, password, key, vault=vault)
        except DecryptionError as e:
            raise RuntimeError(f"Failed to decrypt {key}: {e}") from e

    _SECRETS_INITIALIZED = True


def reset_secrets_initialization() -> None:
    """Reset the initialization flag. Useful for testing."""
    global _SECRETS_INITIALIZED
    _SECRETS_INITIALIZED = False


class ToolFailure(Exception):
    """Exception to indicate a tool execution failure."""
    def __init__(self, message: str, *, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class ToolResult(BaseModel):
    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    def __bool__(self):
        fields = type(self).model_fields
        return any(getattr(self, attr) for attr in fields.keys())

    def __add__(self, other: "ToolResult") -> "ToolResult":
        def combine_fields(field: Optional[str], other_field: Optional[str], concatenate: bool = False):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot concatenate non-string fields")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self) -> str:
        return f"Error: {self.error}" if self.error else f"Output: {self.output}"

    def replace(self, **kwargs) -> "ToolResult":
        return type(self)(**{**self.model_dump(), **kwargs})
