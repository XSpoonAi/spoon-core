"""TodoList Middleware - Task Planning and Progress Tracking.

Provides todo list tools to agents for structured task management:
- write_todos: Create/update todo list with tasks
- read_todos: Read current todo list state

Compatible with LangChain DeepAgents TodoListMiddleware interface.

Usage:
    from spoon_ai.middleware.todolist import TodoListMiddleware

    agent = ToolCallAgent(
        middleware=[TodoListMiddleware()],
        ...
    )
"""

import json
import logging
from typing import Any, Callable, Dict, List, Literal, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResult,
)
from spoon_ai.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ============================================================================
# Todo Data Types
# ============================================================================

class TodoStatus(str, Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class TodoItem:
    """A single todo item."""
    content: str
    status: TodoStatus = TodoStatus.PENDING
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status.value,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        return cls(
            content=data["content"],
            status=TodoStatus(data.get("status", "pending")),
            id=data.get("id"),
        )


@dataclass
class TodoList:
    """Container for todo items."""
    todos: List[TodoItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "todos": [t.to_dict() for t in self.todos]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoList":
        todos = [TodoItem.from_dict(t) for t in data.get("todos", [])]
        return cls(todos=todos)

    def format_display(self) -> str:
        """Format todo list for display."""
        if not self.todos:
            return "No todos in list."

        lines = ["Current Todo List:", ""]
        for i, todo in enumerate(self.todos, 1):
            status_icon = {
                TodoStatus.PENDING: "[ ]",
                TodoStatus.IN_PROGRESS: "[~]",
                TodoStatus.COMPLETED: "[x]",
            }.get(todo.status, "[ ]")
            lines.append(f"{i}. {status_icon} {todo.content}")

        # Summary
        pending = sum(1 for t in self.todos if t.status == TodoStatus.PENDING)
        in_progress = sum(1 for t in self.todos if t.status == TodoStatus.IN_PROGRESS)
        completed = sum(1 for t in self.todos if t.status == TodoStatus.COMPLETED)
        lines.append("")
        lines.append(f"Summary: {completed} completed, {in_progress} in progress, {pending} pending")

        return "\n".join(lines)


# ============================================================================
# Tool Descriptions
# ============================================================================

WRITE_TODOS_DESCRIPTION = """Create or update your todo list for the current task.

Use this tool to:
- Plan complex multi-step tasks
- Track progress through a workflow
- Break down large problems into manageable steps

Each todo should have:
- content: Description of what needs to be done (imperative form, e.g., "Fix the bug")
- status: "pending", "in_progress", or "completed"

Best practices:
- Create todos at the start of complex tasks
- Mark todos as "in_progress" when you start working on them
- Mark todos as "completed" immediately when done
- Keep only ONE todo as "in_progress" at a time
- Remove irrelevant todos rather than leaving them pending

When NOT to use:
- Simple single-step tasks
- Pure research or information gathering
- Tasks that can be completed in one action"""

READ_TODOS_DESCRIPTION = """Read your current todo list.

Returns the current state of all todos with their status:
- [ ] pending: Not yet started
- [~] in_progress: Currently working on
- [x] completed: Finished

Use this to:
- Check what tasks remain
- Review progress before next action
- Decide which task to work on next"""


# ============================================================================
# Todo Tools
# ============================================================================

class WriteTodosTool(BaseTool):
    """Tool to create/update todo list."""

    name: str = "write_todos"
    description: str = WRITE_TODOS_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of todo items",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Description of the task"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Task status",
                            "default": "pending"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        "required": ["todos"]
    }
    _middleware: Any = None

    def __init__(self, middleware: "TodoListMiddleware", **kwargs):
        super().__init__(**kwargs)
        self._middleware = middleware

    async def execute(self, todos: List[Dict[str, Any]], **kwargs) -> str:
        """Update the todo list."""
        # Convert to TodoItem objects
        todo_items = []
        for i, t in enumerate(todos):
            item = TodoItem(
                content=t.get("content", ""),
                status=TodoStatus(t.get("status", "pending")),
                id=str(i + 1)
            )
            todo_items.append(item)

        # Update middleware state
        self._middleware._todo_list = TodoList(todos=todo_items)

        # Format response
        return f"Todo list updated with {len(todo_items)} items.\n\n{self._middleware._todo_list.format_display()}"


class ReadTodosTool(BaseTool):
    """Tool to read current todo list."""

    name: str = "read_todos"
    description: str = READ_TODOS_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": []
    }
    _middleware: Any = None

    def __init__(self, middleware: "TodoListMiddleware", **kwargs):
        super().__init__(**kwargs)
        self._middleware = middleware

    async def execute(self, **kwargs) -> str:
        """Read the current todo list."""
        return self._middleware._todo_list.format_display()


# ============================================================================
# System Prompt
# ============================================================================

TODOLIST_SYSTEM_PROMPT = """## Task Management

You have access to a todo list for tracking progress through complex tasks:

- write_todos: Create or update your todo list
- read_todos: Read current todo list state

### When to use todos:
1. Multi-step tasks requiring 3+ distinct actions
2. Complex problems needing careful planning
3. User provides multiple tasks or requirements

### When NOT to use todos:
1. Simple single-step tasks
2. Quick questions or lookups
3. Tasks completable in one action

### Best practices:
- Create todos BEFORE starting complex work
- Keep exactly ONE todo as "in_progress" at a time
- Mark todos "completed" IMMEDIATELY when done
- Don't create todos just for the sake of it"""


# ============================================================================
# TodoList Middleware
# ============================================================================

class TodoListMiddleware(AgentMiddleware):
    """Middleware for providing todo list tools to an agent.

    Provides two tools:
    - write_todos: Create/update todo list
    - read_todos: Read current todo list

    Example:
        ```python
        from spoon_ai.middleware.todolist import TodoListMiddleware

        middleware = TodoListMiddleware()

        agent = ToolCallAgent(
            middleware=[middleware],
            ...
        )
        ```
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        auto_inject_prompt: bool = True,
    ):
        """Initialize TodoList middleware.

        Args:
            system_prompt: Optional custom system prompt override.
            auto_inject_prompt: Whether to auto-inject system prompt (default: True)
        """
        super().__init__()

        self._custom_system_prompt = system_prompt
        self._auto_inject_prompt = auto_inject_prompt
        self._todo_list = TodoList()

        # Create tools
        self._tools = [
            WriteTodosTool(middleware=self),
            ReadTodosTool(middleware=self),
        ]

    @property
    def tools(self) -> List[BaseTool]:
        """Get todo list tools."""
        return self._tools

    @property
    def system_prompt(self) -> str:
        """Get system prompt for todo list tools."""
        if self._custom_system_prompt:
            return self._custom_system_prompt
        return TODOLIST_SYSTEM_PROMPT

    @property
    def todo_list(self) -> TodoList:
        """Get current todo list."""
        return self._todo_list

    def get_todos_state(self) -> Dict[str, Any]:
        """Get todo list as state dict (for checkpointing)."""
        return self._todo_list.to_dict()

    def restore_todos_state(self, state: Dict[str, Any]) -> None:
        """Restore todo list from state dict."""
        self._todo_list = TodoList.from_dict(state)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable
    ) -> ModelResponse:
        """Inject system prompt for todo list tools."""
        if not self._auto_inject_prompt:
            return await handler(request)

        # Append todo list system prompt
        if request.system_prompt:
            new_prompt = f"{request.system_prompt}\n\n{self.system_prompt}"
        else:
            new_prompt = self.system_prompt

        # Create new request with updated system prompt
        request = ModelRequest(
            messages=request.messages,
            system_prompt=new_prompt,
            tools=request.tools,
            phase=request.phase,
            runtime=request.runtime,
        )

        return await handler(request)

    def before_agent(self, state: Dict[str, Any], runtime: AgentRuntime) -> Optional[Dict[str, Any]]:
        """Restore todo list from agent state if available."""
        if "todos" in state:
            self.restore_todos_state(state["todos"])
            logger.debug(f"Restored {len(self._todo_list.todos)} todos from state")
        return None

    def after_agent(self, state: Dict[str, Any], runtime: AgentRuntime) -> Optional[Dict[str, Any]]:
        """Save todo list to agent state."""
        return {"todos": self.get_todos_state()}
