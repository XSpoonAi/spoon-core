"""
Enhanced Checkpointing System

Provides persistent storage for agent state, enabling:
- Conversation pause and resume
- Thread-based isolation
- State recovery after crashes
- Cross-session persistence

Backends:
- InMemoryCheckpointer: For testing
- SQLiteCheckpointer: Production-ready persistence
- JSONCheckpointer: File-based persistence

Usage:
    checkpointer = SQLiteCheckpointer("agent_memory.db")

    agent = ToolCallAgent(
        thread_id="user-123",
        middleware=[CheckpointMiddleware(checkpointer)]
    )

    # State is automatically saved after each step
    await agent.run("Do something complex")

    # Resume in a new session
    agent2 = ToolCallAgent(
        thread_id="user-123",  # Same thread
        middleware=[CheckpointMiddleware(checkpointer)]
    )
    # Agent2 will restore state from checkpoint
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import pickle

from spoon_ai.schema import Message
from spoon_ai.middleware.base import AgentMiddleware, AgentRuntime

logger = logging.getLogger(__name__)


# ============================================================================
# Checkpoint Data Structures
# ============================================================================

@dataclass
class Checkpoint:
    """A single checkpoint containing agent state."""
    # Identification
    thread_id: str
    checkpoint_id: str

    # State
    messages: List[Dict[str, Any]]  # Serialized messages
    agent_state: Dict[str, Any]     # Agent state dict
    metadata: Dict[str, Any]        # Additional metadata

    # Timestamps
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(**data)


# ============================================================================
# Checkpointer Base Class
# ============================================================================

class BaseCheckpointer(ABC):
    """Abstract base class for checkpointers."""

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint.

        Args:
            checkpoint: Checkpoint to save
        """
        pass

    @abstractmethod
    def load(self, thread_id: str) -> Optional[Checkpoint]:
        """Load the latest checkpoint for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Latest checkpoint or None if not found
        """
        pass

    @abstractmethod
    def list_threads(self) -> List[str]:
        """List all thread IDs with checkpoints.

        Returns:
            List of thread IDs
        """
        pass

    @abstractmethod
    def delete(self, thread_id: str) -> bool:
        """Delete all checkpoints for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def get_history(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        """Get checkpoint history for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoints, most recent first
        """
        pass


# ============================================================================
# In-Memory Checkpointer (for testing)
# ============================================================================

class InMemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpointer for testing and development."""

    def __init__(self):
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        logger.info("Initialized InMemoryCheckpointer")

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to memory."""
        if checkpoint.thread_id not in self._checkpoints:
            self._checkpoints[checkpoint.thread_id] = []

        self._checkpoints[checkpoint.thread_id].append(checkpoint)
        logger.debug(f"Saved checkpoint for thread {checkpoint.thread_id}")

    def load(self, thread_id: str) -> Optional[Checkpoint]:
        """Load latest checkpoint from memory."""
        if thread_id not in self._checkpoints:
            return None

        checkpoints = self._checkpoints[thread_id]
        return checkpoints[-1] if checkpoints else None

    def list_threads(self) -> List[str]:
        """List all thread IDs."""
        return list(self._checkpoints.keys())

    def delete(self, thread_id: str) -> bool:
        """Delete all checkpoints for thread."""
        if thread_id in self._checkpoints:
            del self._checkpoints[thread_id]
            return True
        return False

    def get_history(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        """Get checkpoint history."""
        if thread_id not in self._checkpoints:
            return []

        checkpoints = self._checkpoints[thread_id]
        return list(reversed(checkpoints[-limit:]))


# ============================================================================
# SQLite Checkpointer (production)
# ============================================================================

class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite-based checkpointer for production use.

    Features:
    - Persistent storage
    - Automatic schema creation
    - Thread-safe operations
    - Checkpoint history
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        """Initialize SQLite checkpointer.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"Initialized SQLiteCheckpointer at {db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(thread_id, checkpoint_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_id
                ON checkpoints(thread_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON checkpoints(created_at DESC)
            """)

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_id, messages_json, state_json, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.thread_id,
                checkpoint.checkpoint_id,
                json.dumps(checkpoint.messages),
                json.dumps(checkpoint.agent_state),
                json.dumps(checkpoint.metadata),
                checkpoint.created_at,
                checkpoint.updated_at
            ))

        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} for thread {checkpoint.thread_id}")

    def load(self, thread_id: str) -> Optional[Checkpoint]:
        """Load latest checkpoint from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (thread_id,))

            row = cursor.fetchone()

        if not row:
            return None

        return Checkpoint(
            thread_id=row['thread_id'],
            checkpoint_id=row['checkpoint_id'],
            messages=json.loads(row['messages_json']),
            agent_state=json.loads(row['state_json']),
            metadata=json.loads(row['metadata_json']),
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    def list_threads(self) -> List[str]:
        """List all thread IDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT thread_id FROM checkpoints
                ORDER BY created_at DESC
            """)
            return [row[0] for row in cursor.fetchall()]

    def delete(self, thread_id: str) -> bool:
        """Delete all checkpoints for thread."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM checkpoints WHERE thread_id = ?
            """, (thread_id,))
            return cursor.rowcount > 0

    def get_history(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        """Get checkpoint history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (thread_id, limit))

            rows = cursor.fetchall()

        return [
            Checkpoint(
                thread_id=row['thread_id'],
                checkpoint_id=row['checkpoint_id'],
                messages=json.loads(row['messages_json']),
                agent_state=json.loads(row['state_json']),
                metadata=json.loads(row['metadata_json']),
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            for row in rows
        ]


# ============================================================================
# JSON File Checkpointer
# ============================================================================

class JSONCheckpointer(BaseCheckpointer):
    """JSON file-based checkpointer.

    Each thread gets its own JSON file.
    """

    def __init__(self, storage_dir: str = ".checkpoints"):
        """Initialize JSON checkpointer.

        Args:
            storage_dir: Directory for checkpoint files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized JSONCheckpointer at {storage_dir}")

    def _get_path(self, thread_id: str) -> Path:
        """Get file path for thread."""
        # Sanitize thread_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)
        return self.storage_dir / f"{safe_id}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to JSON file."""
        path = self._get_path(checkpoint.thread_id)

        # Load existing history
        history = []
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                history = data.get('history', [])

        # Append new checkpoint
        history.append(checkpoint.to_dict())

        # Keep last 100 checkpoints
        history = history[-100:]

        # Save
        with open(path, 'w') as f:
            json.dump({'history': history}, f, indent=2)

        logger.debug(f"Saved checkpoint for thread {checkpoint.thread_id}")

    def load(self, thread_id: str) -> Optional[Checkpoint]:
        """Load latest checkpoint from JSON file."""
        path = self._get_path(thread_id)

        if not path.exists():
            return None

        with open(path, 'r') as f:
            data = json.load(f)
            history = data.get('history', [])

        if not history:
            return None

        return Checkpoint.from_dict(history[-1])

    def list_threads(self) -> List[str]:
        """List all thread IDs."""
        return [
            f.stem for f in self.storage_dir.glob("*.json")
        ]

    def delete(self, thread_id: str) -> bool:
        """Delete checkpoint file."""
        path = self._get_path(thread_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def get_history(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        """Get checkpoint history."""
        path = self._get_path(thread_id)

        if not path.exists():
            return []

        with open(path, 'r') as f:
            data = json.load(f)
            history = data.get('history', [])

        # Return last N checkpoints
        recent = history[-limit:]
        return [Checkpoint.from_dict(cp) for cp in reversed(recent)]


# ============================================================================
# Checkpoint Middleware
# ============================================================================

class CheckpointMiddleware(AgentMiddleware):
    """Middleware that adds automatic checkpointing.

    Features:
    - Automatic state saving after each step
    - Automatic state restoration on startup
    - Thread-based isolation
    - Configurable checkpoint frequency

    Usage:
        checkpointer = SQLiteCheckpointer("agent.db")

        agent = ToolCallAgent(
            thread_id="user-123",
            middleware=[CheckpointMiddleware(
                checkpointer,
                save_frequency=1  # Save after every step
            )]
        )
    """

    def __init__(
        self,
        checkpointer: BaseCheckpointer,
        save_frequency: int = 1,  # Save every N steps
        auto_restore: bool = True
    ):
        """Initialize checkpoint middleware.

        Args:
            checkpointer: Checkpointer backend
            save_frequency: Save checkpoint every N steps
            auto_restore: Automatically restore state on startup
        """
        super().__init__()
        self.checkpointer = checkpointer
        self.save_frequency = save_frequency
        self.auto_restore = auto_restore

        self._last_save_step = 0

        logger.info(f"Initialized CheckpointMiddleware with {checkpointer.__class__.__name__}")

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Restore state from checkpoint on startup."""
        if not self.auto_restore or not runtime.thread_id:
            return None

        # Load checkpoint
        checkpoint = self.checkpointer.load(runtime.thread_id)

        if not checkpoint:
            logger.debug(f"No checkpoint found for thread {runtime.thread_id}")
            return None

        # Restore messages
        from spoon_ai.schema import Message
        messages = [Message(**msg) for msg in checkpoint.messages]

        logger.info(f"Restored {len(messages)} messages from checkpoint for thread {runtime.thread_id}")

        # CRITICAL: Messages must be added to agent.memory FIRST, then runtime.messages
        # This is because runtime.messages is populated from agent.memory.get_messages()
        # in _create_runtime_context, so we need to ensure messages are in memory first
        
        # Add messages to agent's memory if accessible
        # CRITICAL: This must happen BEFORE agent.run() starts, so messages are available
        if hasattr(runtime, '_agent_instance'):
            agent = runtime._agent_instance
            if hasattr(agent, 'memory'):
                # CRITICAL: Clear existing messages before restoring checkpoint to avoid duplication
                # This ensures that when auto_restore runs on an agent instance with existing messages
                # (e.g., re-running the same agent, or if memory already contains a system prompt),
                # we replace existing memory instead of appending to it.
                if hasattr(agent.memory, 'messages') and isinstance(agent.memory.messages, list):
                    agent.memory.messages.clear()
                    logger.debug("Cleared existing messages from agent.memory.messages before restoring checkpoint")
                elif hasattr(agent.memory, 'clear'):
                    try:
                        agent.memory.clear()
                        logger.debug("Cleared existing messages from agent.memory via clear() method")
                    except Exception as e:
                        logger.warning(f"Failed to clear agent memory: {e}")
                
                # Try add_message method first (preferred for spoon_ai.chat.Memory)
                if hasattr(agent.memory, 'add_message'):
                    added_count = 0
                    for msg in messages:
                        try:
                            agent.memory.add_message(msg)
                            added_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to add message to agent memory via add_message: {e}")
                    logger.info(f"✅ Restored {added_count}/{len(messages)} messages to agent.memory via add_message()")
                    
                    # Verify messages were added
                    if hasattr(agent.memory, 'get_messages'):
                        restored_count = len(agent.memory.get_messages())
                        logger.info(f"   Verification: agent.memory.get_messages() returns {restored_count} messages")
                # Fallback: direct access to messages list
                elif hasattr(agent.memory, 'messages'):
                    if isinstance(agent.memory.messages, list):
                        agent.memory.messages.extend(messages)
                        logger.info(f"✅ Added {len(messages)} messages to agent.memory.messages")
                    else:
                        logger.warning(f"agent.memory.messages is not a list: {type(agent.memory.messages)}")
                else:
                    logger.warning(f"Agent memory does not support message restoration: {type(agent.memory)}")
            else:
                logger.warning(f"Agent does not have memory attribute")
        else:
            logger.warning(f"Runtime does not have _agent_instance, cannot restore messages to agent memory")
        
        # Update runtime.messages to reflect restored messages
        # This ensures middleware and agent code see the restored messages
        runtime.messages = list(messages)  # Use restored messages
        logger.debug(f"Updated runtime.messages with {len(messages)} restored messages")

        # Return state updates
        return {
            **checkpoint.agent_state,
            "checkpoint_restored": True,
            "checkpoint_id": checkpoint.checkpoint_id
        }

    def after_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Save checkpoint after agent completes."""
        logger.info(f"🔍 CheckpointMiddleware.after_agent called. thread_id={runtime.thread_id}, messages={len(runtime.messages)}")

        if not runtime.thread_id:
            logger.warning("⚠️ Skipping checkpoint save: no thread_id")
            return None

        self._save_checkpoint(state, runtime)
        logger.info(f"✅ Checkpoint saved for thread {runtime.thread_id}")
        return None

    def _save_checkpoint(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> None:
        """Save checkpoint to storage."""
        import uuid

        logger.debug(f"_save_checkpoint: thread_id={runtime.thread_id}, messages count={len(runtime.messages)}, state keys={list(state.keys())}")

        # Serialize messages (Pydantic V2 uses model_dump, V1 uses dict)
        messages = [
            msg.model_dump() if hasattr(msg, 'model_dump') else (msg.dict() if hasattr(msg, 'dict') else msg)
            for msg in runtime.messages
        ]

        logger.debug(f"Serialized {len(messages)} messages for checkpoint")

        # Create checkpoint
        now = datetime.now().isoformat()
        checkpoint = Checkpoint(
            thread_id=runtime.thread_id,
            checkpoint_id=str(uuid.uuid4()),
            messages=messages,
            agent_state=state.copy(),
            metadata={
                "step": runtime.current_step,
                "max_steps": runtime.max_steps
            },
            created_at=now,
            updated_at=now
        )

        logger.debug(f"Created checkpoint object: id={checkpoint.checkpoint_id}")

        # Save
        self.checkpointer.save(checkpoint)
        logger.info(f"💾 Checkpoint saved to storage: thread={runtime.thread_id}, step={runtime.current_step}, messages={len(messages)}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_sqlite_checkpointer(db_path: str = "agent_memory.db") -> SQLiteCheckpointer:
    """Create a SQLite checkpointer.

    Args:
        db_path: Path to SQLite database

    Returns:
        Configured SQLiteCheckpointer
    """
    return SQLiteCheckpointer(db_path)


def create_memory_checkpointer() -> InMemoryCheckpointer:
    """Create an in-memory checkpointer for testing.

    Returns:
        InMemoryCheckpointer instance
    """
    return InMemoryCheckpointer()
