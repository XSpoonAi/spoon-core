"""
spoon_ai.graph package

Public facade for the graph engine. Import from here.
"""
# Re-export public API
from .exceptions import (
    GraphExecutionError,
    NodeExecutionError,
    StateValidationError,
    CheckpointError,
    GraphConfigurationError,
    EdgeRoutingError,
    InterruptError,
)
from .types import (
    NodeContext,
    NodeResult,
    RouterResult,
    ParallelBranchConfig,
    Command,
    StateSnapshot,
    CheckpointTuple
)
from .reducers import (
    add_messages,
    merge_dicts,
    append_history,
    union_sets,
    validate_range,
    validate_enum,
)
from .decorators import (
    node_decorator,
    router_decorator,
)
from .checkpointer import InMemoryCheckpointer
from .cache import (
    BaseCache,
    InMemoryCache,
    SQLiteCache,
    CacheEntry,
    compute_cache_key,
    create_memory_cache,
    create_sqlite_cache,
)

# Engine and agent implementations (now within this package)
from .engine import (
    StateGraph,
    CompiledGraph,
    BaseNode,
    RunnableNode,
    ToolNode,
    ConditionNode,
    START,
    END,
    interrupt,
    # Multimodal utilities for graph workflows
    create_multimodal_message,
    create_vision_user_message,
    create_pdf_message,
    create_document_message,
)
from .agent import GraphAgent, AgentStateCheckpoint, MockMemory, Memory
