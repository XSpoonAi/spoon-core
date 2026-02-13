from __future__ import annotations

from pathlib import Path
from .security import init_security
try:  # Python 3.12+
    from importlib.metadata import PackageNotFoundError, version as _dist_version
except ImportError:  # pragma: no cover - fallback for older environments
    try:
        from importlib_metadata import (  # type: ignore
            PackageNotFoundError,
            version as _dist_version,
        )
    except ImportError:  # Last-resort placeholders
        PackageNotFoundError = Exception  # type: ignore


def _read_local_pyproject_version() -> str | None:
    """Attempt to read version from local pyproject when running from source.

    Returns None if pyproject.toml is missing or cannot be parsed.
    """
    root = Path(__file__).resolve().parents[1]
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    try:
        try:
            import tomllib  # Python 3.12+
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore

        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        project = data.get("project", {}) if isinstance(data, dict) else {}
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except Exception:
        return None
    return None


def _resolve_version() -> str:
    # Prefer installed distribution metadata when available
    try:
        return _dist_version("spoon-ai-sdk")
    except PackageNotFoundError:
        pass
    except Exception:
        pass

    # Fallback: read from local pyproject when running from source
    local = _read_local_pyproject_version()
    if isinstance(local, str) and local:
        return local

    # Final fallback
    return "0.0.0"


__version__: str = _resolve_version()

# Optional runtime imports: make top-level import of spoon_ai lightweight
# so that subpackages (e.g., spoon_ai.rag) can be used in minimal/offline envs.
try:
    from spoon_ai.chat import ChatBot  # type: ignore
    from spoon_ai.schema import (  # type: ignore
        LLMResponse, LLMResponseChunk, Message,
        # Multimodal content types
        TextContent, ImageContent, ImageUrlContent, DocumentContent,
        ImageSource, ImageUrlSource, DocumentSource, MessageContent, ContentBlock
    )
    _exported = [
        "__version__",
        "ChatBot",
        "Message",
        "LLMResponse",
        "LLMResponseChunk",
        # Multimodal exports
        "TextContent",
        "ImageContent",
        "ImageUrlContent",
        "DocumentContent",
        "ImageSource",
        "ImageUrlSource",
        "DocumentSource",
        "MessageContent",
        "ContentBlock",
    ]
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency surface
    ChatBot = None  # type: ignore
    Message = None  # type: ignore
    LLMResponse = None  # type: ignore
    LLMResponseChunk = None  # type: ignore
    TextContent = None  # type: ignore
    ImageContent = None  # type: ignore
    ImageUrlContent = None  # type: ignore
    DocumentContent = None  # type: ignore
    ImageSource = None  # type: ignore
    ImageUrlSource = None  # type: ignore
    DocumentSource = None  # type: ignore
    MessageContent = None  # type: ignore
    ContentBlock = None  # type: ignore
    _exported = ["__version__"]


init_security()

__all__ = _exported

