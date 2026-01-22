"""
Cache System for Graph Workflows.

Provides caching for node outputs in graph workflows to avoid redundant
computation and speed up execution.

Types of caching:
1. Node-level caching - caches node outputs based on inputs
2. In-memory and persistent (SQLite) backends

Compatible with LangGraph BaseCache interface.

Usage:
    from spoon_ai.graph.cache import InMemoryCache, SQLiteCache

    # In-memory cache (for testing/short sessions)
    cache = InMemoryCache()

    # SQLite cache (persistent across sessions)
    cache = SQLiteCache("cache.db")

    # Use with graph
    graph = StateGraph(...)
    compiled = graph.compile(cache=cache)
"""

import hashlib
import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Key Utilities
# ============================================================================

def compute_cache_key(
    node_name: str,
    inputs: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Compute a cache key from node name and inputs.

    Args:
        node_name: Name of the node
        inputs: Input values to the node
        config: Optional configuration

    Returns:
        SHA256 hash as cache key
    """
    key_data = {
        "node": node_name,
        "inputs": _serialize_for_hash(inputs),
    }
    if config:
        key_data["config"] = _serialize_for_hash(config)

    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _serialize_for_hash(obj: Any) -> Any:
    """Serialize object for hashing, handling non-JSON types."""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_hash(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_for_hash(v) for k, v in sorted(obj.items())}
    elif hasattr(obj, 'model_dump'):  # Pydantic v2
        return _serialize_for_hash(obj.model_dump())
    elif hasattr(obj, 'dict'):  # Pydantic v1
        return _serialize_for_hash(obj.dict())
    else:
        return str(obj)


# ============================================================================
# Cache Entry
# ============================================================================

class CacheEntry:
    """A cached value with metadata."""

    def __init__(
        self,
        key: str,
        value: Any,
        node_name: str,
        created_at: Optional[datetime] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.key = key
        self.value = value
        self.node_name = node_name
        self.created_at = created_at or datetime.now()
        self.ttl_seconds = ttl_seconds
        self.metadata = metadata or {}

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "node_name": self.node_name,
            "created_at": self.created_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Deserialize from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            node_name=data["node_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            ttl_seconds=data.get("ttl_seconds"),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Base Cache (Abstract)
# ============================================================================

class BaseCache(ABC):
    """Abstract base class for graph caches.

    Compatible with LangGraph BaseCache interface.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        node_name: str = "",
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            node_name: Name of the node that produced this value
            ttl_seconds: Time-to-live in seconds (None = no expiry)
            metadata: Optional metadata
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    def get_or_compute(
        self,
        key: str,
        compute_fn: callable,
        node_name: str = "",
        ttl_seconds: Optional[int] = None,
    ) -> Tuple[Any, bool]:
        """Get cached value or compute and cache it.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            node_name: Name of the node
            ttl_seconds: Time-to-live for cached value

        Returns:
            Tuple of (value, was_cached)
        """
        cached = self.get(key)
        if cached is not None:
            return cached, True

        value = compute_fn()
        self.set(key, value, node_name=node_name, ttl_seconds=ttl_seconds)
        return value, False


# ============================================================================
# In-Memory Cache
# ============================================================================

class InMemoryCache(BaseCache):
    """In-memory cache implementation.

    Fast but not persistent across sessions. Suitable for:
    - Testing
    - Short-running workflows
    - When persistence is not needed
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl_seconds: Optional[int] = None,
    ):
        """Initialize in-memory cache.

        Args:
            max_entries: Maximum number of entries to keep
            default_ttl_seconds: Default TTL for entries (None = no expiry)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        node_name: str = "",
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set cached value."""
        # Evict oldest entries if at capacity
        if len(self._cache) >= self._max_entries and key not in self._cache:
            self._evict_oldest()

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            node_name=node_name,
            ttl_seconds=ttl,
            metadata=metadata,
        )

    def delete(self, key: str) -> bool:
        """Delete cached value."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def _evict_oldest(self) -> None:
        """Evict oldest entry to make room."""
        if not self._cache:
            return

        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        del self._cache[oldest_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


# ============================================================================
# SQLite Cache
# ============================================================================

class SQLiteCache(BaseCache):
    """SQLite-based persistent cache.

    Persistent across sessions. Suitable for:
    - Production use
    - Long-running workflows
    - When you want to reuse cached results
    """

    def __init__(
        self,
        db_path: str = "graph_cache.db",
        default_ttl_seconds: Optional[int] = None,
        max_entries: Optional[int] = None,
    ):
        """Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file
            default_ttl_seconds: Default TTL for entries (None = no expiry)
            max_entries: Maximum entries to keep (None = unlimited)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries

        self._init_db()
        logger.info(f"Initialized SQLiteCache at {db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    node_name TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    metadata_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON cache(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON cache(created_at)
            """)

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        self._cleanup_expired()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT value_json FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return None

    def set(
        self,
        key: str,
        value: Any,
        node_name: str = "",
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set cached value."""
        now = datetime.now()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        expires_at = None
        if ttl is not None:
            from datetime import timedelta
            expires_at = (now + timedelta(seconds=ttl)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache
                (key, value_json, node_name, created_at, expires_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                key,
                json.dumps(value, default=str),
                node_name,
                now.isoformat(),
                expires_at,
                json.dumps(metadata or {}),
            ))

        # Enforce max entries if set
        if self._max_entries:
            self._enforce_max_entries()

    def delete(self, key: str) -> bool:
        """Delete cached value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE key = ?",
                (key,)
            )
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all cached values."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")

    def _enforce_max_entries(self) -> None:
        """Remove oldest entries if over limit."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]

            if count > self._max_entries:
                to_delete = count - self._max_entries
                conn.execute("""
                    DELETE FROM cache WHERE key IN (
                        SELECT key FROM cache
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                """, (to_delete,))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (datetime.now().isoformat(),)
            )
            expired = cursor.fetchone()[0]

        return {
            "entries": count,
            "expired_pending_cleanup": expired,
            "db_path": str(self.db_path),
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_memory_cache(
    max_entries: int = 1000,
    default_ttl_seconds: Optional[int] = None,
) -> InMemoryCache:
    """Create an in-memory cache.

    Args:
        max_entries: Maximum entries to keep
        default_ttl_seconds: Default TTL

    Returns:
        Configured InMemoryCache
    """
    return InMemoryCache(
        max_entries=max_entries,
        default_ttl_seconds=default_ttl_seconds,
    )


def create_sqlite_cache(
    db_path: str = "graph_cache.db",
    default_ttl_seconds: Optional[int] = None,
    max_entries: Optional[int] = None,
) -> SQLiteCache:
    """Create a SQLite cache.

    Args:
        db_path: Path to database file
        default_ttl_seconds: Default TTL
        max_entries: Maximum entries

    Returns:
        Configured SQLiteCache
    """
    return SQLiteCache(
        db_path=db_path,
        default_ttl_seconds=default_ttl_seconds,
        max_entries=max_entries,
    )
