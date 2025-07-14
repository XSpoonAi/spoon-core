from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Function(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function

class AgentState(str, Enum):
    """
    The state of the agent.
    """
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

class ToolChoice(str, Enum):
    """Tool choice options"""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class Role(str, Enum):
    """Message role options"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...) # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES] # type: ignore

class LLMResponse(BaseModel):
    content: str
    tool_calls: List[Any] = Field(default_factory=list)
    finish_reason: Optional[str] = Field(default=None)
    native_finish_reason: Optional[str] = Field(default=None)
