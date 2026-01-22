"""
Middleware System for Deep Agents

Provides flexible middleware architecture for:
- Plan-Act-Reflect execution cycles
- Model and tool call interception
- Agent lifecycle hooks
- Declarative tool and state injection
- Filesystem operations with 7 built-in tools
- Todo list task tracking
- Context summarization
- Dangling tool call patching
- Anthropic prompt caching
"""

from .base import (
    # Core middleware classes
    AgentMiddleware,
    MiddlewarePipeline,
    create_middleware_pipeline,

    # Execution phases
    AgentPhase,

    # Runtime context
    AgentRuntime,

    # Request/Response types
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResult,

    # Protocols
    PhaseHook,
)

from .filesystem import (
    FilesystemMiddleware,
    create_filesystem_middleware,
    create_sandbox_backend,
    LocalSandboxBackend,
    # Individual tools
    LsTool,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    ExecuteTool,
    get_filesystem_tools,
)

from .todolist import (
    TodoListMiddleware,
    TodoItem,
    TodoList,
    TodoStatus,
    WriteTodosTool,
    ReadTodosTool,
)

from .summarization import (
    SummarizationMiddleware,
    create_summarization_middleware,
    # Context size types
    ContextSize,
    ContextFraction,
    ContextTokens,
    ContextMessages,
    # Token counting
    TokenCounter,
    count_tokens_approximately,
    # Message removal
    RemoveMessage,
    REMOVE_ALL_MESSAGES,
    # Prompt
    DEFAULT_SUMMARY_PROMPT,
)

from .patch_tool_calls import (
    PatchToolCallsMiddleware,
    create_patch_tool_calls_middleware,
)

from .prompt_caching import (
    AnthropicPromptCachingMiddleware,
    create_prompt_caching_middleware,
)

from .planning import (
    PlanningMiddleware,
    create_planning_middleware,
    Plan,
    PlanStep,
)

__all__ = [
    # Core classes
    "AgentMiddleware",
    "MiddlewarePipeline",
    "create_middleware_pipeline",

    # Phases
    "AgentPhase",

    # Runtime
    "AgentRuntime",

    # Types
    "ModelRequest",
    "ModelResponse",
    "ToolCallRequest",
    "ToolCallResult",
    "PhaseHook",

    # Filesystem
    "FilesystemMiddleware",
    "create_filesystem_middleware",
    "create_sandbox_backend",
    "LocalSandboxBackend",
    "LsTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "ExecuteTool",
    "get_filesystem_tools",

    # TodoList
    "TodoListMiddleware",
    "TodoItem",
    "TodoList",
    "TodoStatus",
    "WriteTodosTool",
    "ReadTodosTool",

    # Summarization
    "SummarizationMiddleware",
    "create_summarization_middleware",
    "ContextSize",
    "ContextFraction",
    "ContextTokens",
    "ContextMessages",
    "TokenCounter",
    "count_tokens_approximately",
    "RemoveMessage",
    "REMOVE_ALL_MESSAGES",
    "DEFAULT_SUMMARY_PROMPT",

    # PatchToolCalls
    "PatchToolCallsMiddleware",
    "create_patch_tool_calls_middleware",

    # PromptCaching
    "AnthropicPromptCachingMiddleware",
    "create_prompt_caching_middleware",

    # Planning
    "PlanningMiddleware",
    "create_planning_middleware",
    "Plan",
    "PlanStep",
]
