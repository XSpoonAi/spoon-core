"""
Middleware System for Deep Agents

Provides flexible middleware architecture for:
- Plan-Act-Reflect execution cycles
- Model and tool call interception
- Agent lifecycle hooks
- Declarative tool and state injection
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
]
