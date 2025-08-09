"""
Graph-based agent implementation for SpoonOS.

This module provides the GraphAgent class that executes StateGraph workflows,
integrating the graph execution system with the existing agent architecture.
"""

import logging
import asyncio
import time
from typing import Optional, Dict, Any
from pydantic import Field, validator

from spoon_ai.agents.base import BaseAgent
from spoon_ai.graph import StateGraph, CompiledGraph, GraphExecutionError
from spoon_ai.schema import AgentState

logger = logging.getLogger(__name__)


class GraphAgent(BaseAgent):
    """
    An agent that executes StateGraph workflows.

    This agent provides a bridge between the existing SpoonOS agent architecture
    and the new graph-based execution system. It allows complex, stateful workflows
    to be defined as graphs and executed with proper state management.

    Key Features:
    - Executes StateGraph workflows
    - Maintains compatibility with existing agent interfaces
    - Provides detailed execution logging and error handling
    - Supports both sync and async node functions
    """

    graph: StateGraph = Field(..., description="The StateGraph to execute")
    compiled_graph: Optional[CompiledGraph] = Field(None, description="Compiled graph instance")
    initial_state: Dict[str, Any] = Field(default_factory=dict, description="Initial state for graph execution")
    preserve_state: bool = Field(default=False, description="Whether to preserve state between runs")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from last execution")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """
        Initialize the GraphAgent.

        Args:
            graph: StateGraph instance to execute
            **kwargs: Additional arguments passed to BaseAgent

        Raises:
            ValueError: If no graph is provided
        """
        if "graph" not in kwargs:
            raise ValueError("A StateGraph instance must be provided via 'graph' parameter")

        super().__init__(**kwargs)

        # Compile the graph during initialization
        try:
            self.compiled_graph = self.graph.compile()
            logger.info(f"GraphAgent '{self.name}' initialized with compiled graph")
        except Exception as e:
            logger.error(f"Failed to compile graph for agent '{self.name}': {str(e)}")
            raise ValueError(f"Graph compilation failed: {str(e)}") from e

    @validator('graph')
    def validate_graph(cls, v):
        """Validate that the provided graph is a StateGraph instance."""
        if not isinstance(v, StateGraph):
            raise ValueError("graph must be a StateGraph instance")
        return v

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the graph workflow with robust error handling and state management."""
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Agent {self.name} is not in the IDLE state (currently: {self.state})")

        logger.info(f"GraphAgent '{self.name}' starting execution")
        
        # Store original state for recovery
        original_memory_messages = self.memory.get_messages().copy()
        
        try:
            # Prepare initial state
            execution_state = self._prepare_execution_state(request)
            
            async with self.state_context(AgentState.RUNNING):
                # Execute the compiled graph with timeout
                try:
                    final_state = await asyncio.wait_for(
                        self.compiled_graph.invoke(execution_state), 
                        timeout=300  # 5 minute timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"GraphAgent '{self.name}' execution timed out")
                    raise RuntimeError("Graph execution timed out after 5 minutes")
                
                # Validate final state
                if not self._validate_final_state(final_state):
                    logger.error(f"GraphAgent '{self.name}' produced invalid final state")
                    raise RuntimeError("Graph execution produced invalid final state")
                
                # Store execution metadata
                self.execution_metadata = {
                    "execution_history": self.compiled_graph.get_execution_history(),
                    "final_state_keys": list(final_state.keys()),
                    "execution_successful": True,
                    "execution_time": time.time()
                }

                # Preserve state if enabled
                if self.preserve_state:
                    self._last_state = self._sanitize_preserved_state(final_state)

                # Extract output with validation
                output = self._extract_output(final_state)
                
                # Validate output before adding to memory
                if not output or not isinstance(output, str):
                    logger.warning(f"GraphAgent '{self.name}' produced invalid output: {type(output)}")
                    output = "Graph execution completed but produced invalid output"

                # Add assistant response to memory
                self.add_message("assistant", output)

                logger.info(f"GraphAgent '{self.name}' execution completed successfully")
                return output

        except GraphExecutionError as e:
            logger.error(f"Graph execution failed for agent '{self.name}': {str(e)}")
            self._handle_execution_error(e, original_memory_messages)
            raise RuntimeError(f"Graph execution failed: {str(e)}") from e

        except Exception as e:
            logger.error(f"Unexpected error during graph execution for agent '{self.name}': {str(e)}")
            self._handle_execution_error(e, original_memory_messages)
            raise RuntimeError(f"Graph agent execution failed: {str(e)}") from e

    def _prepare_execution_state(self, request: Optional[str]) -> Dict[str, Any]:
        """Prepare initial execution state with validation."""
        execution_state = {}

        # Start with preserved state if enabled
        if self.preserve_state and hasattr(self, '_last_state'):
            if isinstance(self._last_state, dict):
                execution_state.update(self._last_state)
            else:
                logger.warning(f"GraphAgent '{self.name}' had invalid preserved state type: {type(self._last_state)}")

        # Add configured initial state
        if isinstance(self.initial_state, dict):
            execution_state.update(self.initial_state)

        # Add request to state if provided
        if request is not None:
            execution_state["input"] = request
            execution_state["request"] = request
            self.add_message("user", request)

        # Add messages with validation
        messages = []
        for msg in self.memory.get_messages():
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": getattr(msg, 'tool_calls', None),
                    "tool_call_id": getattr(msg, 'tool_call_id', None)
                })
        
        execution_state["messages"] = messages
        execution_state["agent_name"] = self.name
        execution_state["agent_description"] = self.description
        execution_state["system_prompt"] = self.system_prompt

        return execution_state

    def _validate_final_state(self, final_state: Any) -> bool:
        """Validate that final state is properly formed."""
        if not isinstance(final_state, dict):
            return False
        
        # Should have at least some content
        if not final_state:
            return False
            
        return True

    def _sanitize_preserved_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state for preservation, removing sensitive data."""
        sanitized = state.copy()
        
        # Remove potentially sensitive or large data
        sensitive_keys = ['api_key', 'private_key', 'password', 'token', 'secret']
        for key in list(sanitized.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                del sanitized[key]
        
        return sanitized

    def _handle_execution_error(self, error: Exception, original_messages: list):
        """Handle execution errors with proper cleanup and recovery."""
        # Store error metadata
        self.execution_metadata = {
            "error": str(error),
            "error_type": type(error).__name__,
            "execution_successful": False,
            "execution_time": time.time()
        }
        
        # Reset memory to original state to prevent corruption
        try:
            self.memory.clear()
            for msg in original_messages:
                self.memory.add_message(msg)
            logger.info(f"GraphAgent '{self.name}' memory restored after error")
        except Exception as restore_error:
            logger.error(f"Failed to restore memory after error: {restore_error}")
        
        # Clear any corrupted preserved state
        if hasattr(self, '_last_state'):
            delattr(self, '_last_state')

    async def step(self) -> str:
        """
        Step method for compatibility with BaseAgent.

        Since GraphAgent uses graph execution instead of step-based execution,
        this method is not used in normal operation but is required by the
        BaseAgent interface.

        Returns:
            Status message indicating graph-based execution
        """
        return "GraphAgent uses graph-based execution, not step-based execution"

    def _extract_output(self, final_state: Dict[str, Any]) -> str:
        """Extract the output message from the final graph state with robust validation."""
        if not isinstance(final_state, dict):
            logger.error(f"GraphAgent '{self.name}' received invalid final_state type: {type(final_state)}")
            return "Error: Invalid graph execution result"
        
        # Try different common output keys with validation
        output_keys = ["output", "result", "response", "answer", "final_output", "content"]
        
        for key in output_keys:
            if key in final_state:
                output = final_state[key]
                if output is not None:  # Allow empty strings but not None
                    if isinstance(output, str):
                        return output if output.strip() else f"Graph completed (empty {key})"
                    elif isinstance(output, (dict, list)):
                        try:
                            import json
                            return json.dumps(output, indent=2)
                        except (TypeError, ValueError):
                            return str(output)
                    else:
                        return str(output)
        
        # Message extraction with validation
        if "messages" in final_state and isinstance(final_state["messages"], list):
            assistant_messages = []
            for msg in reversed(final_state["messages"]):
                if (isinstance(msg, dict) and 
                    msg.get("role") == "assistant" and 
                    msg.get("content") and 
                    isinstance(msg.get("content"), str)):
                    assistant_messages.append(msg["content"].strip())
            
            if assistant_messages:
                # Return the most recent non-empty assistant message
                for content in assistant_messages:
                    if content:
                        return content
        
        # Check for error states
        if "error" in final_state:
            error_msg = final_state["error"]
            return f"Graph execution encountered an error: {error_msg}"
        
        # Fallback with state information
        state_keys = list(final_state.keys()) if final_state else []
        if state_keys:
            return f"Graph execution completed. Available state: {', '.join(state_keys[:5])}"
        else:
            logger.warning(f"GraphAgent '{self.name}' completed with empty final state")
            return "Graph execution completed with no output"

    def get_execution_history(self) -> list:
        """
        Get the execution history from the last graph run.

        Returns:
            List of execution steps with metadata
        """
        if self.compiled_graph:
            return self.compiled_graph.get_execution_history()
        return []

    def get_execution_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from the last execution.

        Returns:
            Dictionary containing execution metadata
        """
        return self.execution_metadata.copy()

    def clear_state(self):
        """Clear preserved state and execution history."""
        if hasattr(self, '_last_state'):
            delattr(self, '_last_state')
        self.execution_metadata = {}
        if self.compiled_graph:
            self.compiled_graph.execution_history = []
        logger.debug(f"Cleared state for GraphAgent '{self.name}'")

    def update_initial_state(self, updates: Dict[str, Any]):
        """
        Update the initial state for future executions.

        Args:
            updates: Dictionary of state updates to merge
        """
        self.initial_state.update(updates)
        logger.debug(f"Updated initial state for GraphAgent '{self.name}' with keys: {list(updates.keys())}")

    def set_preserve_state(self, preserve: bool):
        """
        Enable or disable state preservation between runs.

        Args:
            preserve: Whether to preserve state between runs
        """
        self.preserve_state = preserve
        if not preserve and hasattr(self, '_last_state'):
            delattr(self, '_last_state')
        logger.debug(f"Set preserve_state={preserve} for GraphAgent '{self.name}'")