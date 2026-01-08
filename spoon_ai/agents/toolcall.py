import json
import asyncio
import time
import uuid
from logging import getLogger
from typing import Any, List, Optional, Tuple
import logging

from pydantic import AliasChoices, Field
from termcolor import colored

from spoon_ai.agents.react import ReActAgent
from spoon_ai.prompts.toolcall import \
    NEXT_STEP_PROMPT as TOOLCALL_NEXT_STEP_PROMPT
from spoon_ai.prompts.toolcall import SYSTEM_PROMPT as TOOLCALL_SYSTEM_PROMPT
from spoon_ai.schema import TOOL_CHOICE_TYPE, AgentState, ToolCall, ToolChoice, Message, Role
from spoon_ai.tools import ToolManager
from mcp.types import Tool as MCPTool

logging.getLogger("spoon_ai").setLevel(logging.INFO)

logger = getLogger("spoon_ai")


class ToolCallAgent(ReActAgent):

    name: str = "toolcall"
    description: str = "Useful when you need to call a tool"

    system_prompt: str = TOOLCALL_SYSTEM_PROMPT
    next_step_prompt: str = TOOLCALL_NEXT_STEP_PROMPT

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager(tools=[]),
        validation_alias=AliasChoices("available_tools", "avaliable_tools"),
    )
    special_tool_names: List[str] = Field(default_factory=list)

    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore

    tool_calls: List[ToolCall] = Field(default_factory=list)

    output_queue: asyncio.Queue = Field(default_factory=asyncio.Queue)

    # Track last tool error for higher-level fallbacks
    last_tool_error: Optional[str] = Field(default=None, exclude=True)

    # MCP Tools Caching
    mcp_tools_cache: Optional[List[MCPTool]] = Field(default=None, exclude=True)
    mcp_tools_cache_timestamp: Optional[float] = Field(default=None, exclude=True)
    mcp_tools_cache_ttl: float = Field(default=300.0, exclude=True)  # 5 minutes TTL

    def _detect_multimodal_content(self) -> Tuple[bool, bool]:
        """Detect if messages contain images or documents.

        Returns:
            Tuple of (has_images, has_documents)
        """
        has_images = False
        has_documents = False
        try:
            for msg in self.memory.messages:
                if hasattr(msg, 'has_images') and msg.has_images:
                    has_images = True
                if hasattr(msg, 'has_documents') and msg.has_documents:
                    has_documents = True
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if "data:image" in msg.content:
                        has_images = True
                    if "data:application/pdf" in msg.content:
                        has_documents = True
                if has_images and has_documents:
                    break
        except Exception:
            pass
        return has_images, has_documents

    def _has_mcp_tools(self) -> bool:
        """Check if MCP tools are available."""
        try:
            if hasattr(self, 'available_tools') and hasattr(self.available_tools, 'tool_map'):
                if any(hasattr(t, 'mcp_config') for t in self.available_tools.tool_map.values()):
                    return True
            return bool(getattr(self, 'mcp_tools_cache', None))
        except Exception:
            return False

    def _get_subagent_timeout(self) -> float:
        """Calculate timeout for subagent middleware."""
        if hasattr(self, 'middleware') and self.middleware:
            for mw in self.middleware:
                if hasattr(mw, 'subagents') and mw.subagents:
                    max_steps = max((getattr(s, 'max_steps', 5) or 5 for s in mw.subagents), default=5)
                    return max_steps * 120 * 2  # Double buffer
        return 0

    def _calculate_timeout(self, base_timeout: float, for_step: bool = False) -> float:
        """Calculate appropriate timeout based on context.

        Args:
            base_timeout: Base timeout value
            for_step: If True, calculate for step execution; if False, for LLM call
        """
        timeout = base_timeout
        has_images, has_documents = self._detect_multimodal_content()

        # Adjust for multimodal content
        if has_documents:
            timeout = max(timeout, 180.0)  # 3 minutes for PDFs
        elif has_images:
            timeout = max(timeout, 120.0)  # 2 minutes for images

        # Adjust for MCP tools
        if self._has_mcp_tools():
            timeout = max(timeout, 120.0)

        # For step execution, also consider subagents
        if for_step:
            subagent_timeout = self._get_subagent_timeout()
            if subagent_timeout > 0:
                timeout = max(timeout, subagent_timeout)
                logger.info(f"Subagent detected: timeout={timeout}s")

        return timeout

    async def _get_cached_mcp_tools(self) -> List[MCPTool]:
        """Get MCP tools with caching to avoid repeated server calls."""
        current_time = time.time()

        if not hasattr(self, '_cache_lock'):
            self._cache_lock = asyncio.Lock()

        async with self._cache_lock:
            # Check if cache is valid
            if (self.mcp_tools_cache is not None and
                self.mcp_tools_cache_timestamp is not None and
                current_time - self.mcp_tools_cache_timestamp < self.mcp_tools_cache_ttl):
                logger.info(f"♻️ {self.name} using cached MCP tools ({len(self.mcp_tools_cache)} tools)")
                return self.mcp_tools_cache.copy()

            self._invalidate_mcp_cache()

            if hasattr(self, "list_mcp_tools"):
                try:
                    logger.info(f"🔄 {self.name} fetching MCP tools from server...")
                    mcp_tools = await self.list_mcp_tools()

                    if isinstance(mcp_tools, list) and len(mcp_tools) <= 100:
                        self.mcp_tools_cache = mcp_tools
                        self.mcp_tools_cache_timestamp = current_time
                        logger.info(f"📋 {self.name} cached {len(mcp_tools)} MCP tools")
                        return mcp_tools.copy()
                    else:
                        logger.warning(f"⚠️ {self.name} received invalid tools - not caching")
                        return mcp_tools if isinstance(mcp_tools, list) else []
                except Exception as e:
                    logger.error(f"❌ {self.name} failed to fetch MCP tools: {e}")
                    return []

        return []

    def _invalidate_mcp_cache(self):
        """Invalidate MCP tools cache."""
        self.mcp_tools_cache = None
        self.mcp_tools_cache_timestamp = None
        logger.debug(f"🧹 {self.name} invalidated MCP tools cache")

    # Legacy compatibility alias
    @property
    def avaliable_tools(self) -> ToolManager:
        return self.available_tools

    @avaliable_tools.setter
    def avaliable_tools(self, value: ToolManager) -> None:
        self.available_tools = value

    # ========================================================================
    # LLM Call (unified path)
    # ========================================================================

    async def _call_llm(self, tools: list, timeout: float):
        """Unified LLM call - routes through middleware if available."""
        if self._middleware_pipeline:
            return await self._call_llm_with_middleware(tools, timeout)

        return await asyncio.wait_for(
            self.llm.ask_tool(
                messages=self.memory.messages,
                system_msg=self.system_prompt,
                tools=tools,
                tool_choice=self.tool_choices,
                output_queue=self.output_queue,
            ),
            timeout=timeout,
        )

    async def _call_llm_with_middleware(self, tools: list, timeout: float):
        """Call LLM through middleware pipeline."""
        from spoon_ai.middleware.base import ModelRequest, AgentPhase

        runtime = self._create_runtime_context() if hasattr(self, '_create_runtime_context') else None
        tool_choice = self.tool_choices.value if hasattr(self.tool_choices, "value") else self.tool_choices

        request = ModelRequest(
            system_prompt=self.system_prompt,
            messages=self.memory.messages,
            tools=tools,
            tool_choice=tool_choice,
            runtime=runtime,
            phase=AgentPhase.THINK
        )

        async def base_handler(req: ModelRequest):
            return await asyncio.wait_for(
                self.llm.ask_tool(
                    messages=req.messages,
                    system_msg=req.system_prompt,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    output_queue=self.output_queue,
                ),
                timeout=timeout,
            )

        return await self._middleware_pipeline.awrap_model_call(request, base_handler)

    async def think(self) -> bool:
        if self.next_step_prompt:
            await self.add_message("user", self.next_step_prompt)

        # Build tools list
        mcp_tools = await self._get_cached_mcp_tools()
        unique_tools = self._build_tools_list(mcp_tools)

        # Calculate timeout
        base_timeout = max(20.0, min(60.0, getattr(self, '_default_timeout', 30.0) - 5.0))
        llm_timeout = self._calculate_timeout(base_timeout, for_step=False)

        try:
            response = await self._call_llm(unique_tools, llm_timeout)
        except asyncio.TimeoutError:
            logger.error(f"{self.name} LLM tool selection timed out after {llm_timeout}s")
            await self.add_message("assistant", "Tool selection timed out.")
            self.tool_calls = []
            return False

        self.tool_calls = response.tool_calls

        # Check for termination
        if not self.tool_calls and self._should_terminate_on_finish_reason(response):
            logger.info(f"🏁 {self.name} terminating due to finish_reason")
            self.state = AgentState.FINISHED
            await self.add_message("assistant", response.content or "Task completed")
            self._finish_reason_terminated = True
            self._final_response_content = response.content or "Task completed"
            return False

        # Log results
        logger.info(colored(f"🤔 {self.name}'s thoughts received", "cyan"))
        tool_count = len(self.tool_calls) if self.tool_calls else 0
        logger.info(colored(f"🛠️ {self.name} selected {tool_count} tools", "green" if tool_count else "yellow"))

        if self.output_queue:
            self.output_queue.put_nowait({"content": response.content})
            self.output_queue.put_nowait({"tool_calls": response.tool_calls})

        return self._process_think_response(response)

    def _build_tools_list(self, mcp_tools: List[MCPTool]) -> list:
        """Build deduplicated tools list."""
        def convert_mcp_tool(tool: MCPTool) -> dict:
            params = getattr(tool, 'parameters', None) or getattr(tool, 'inputSchema', None) or {
                "type": "object", "properties": {}, "required": []
            }
            return {
                "type": "function",
                "function": {
                    "name": getattr(tool, 'name', 'mcp_tool'),
                    "description": getattr(tool, 'description', 'MCP tool'),
                    "parameters": params
                }
            }

        all_tools = self.available_tools.to_params()
        mcp_tools_params = [convert_mcp_tool(tool) for tool in mcp_tools]

        unique_tools = {}
        for tool in all_tools + mcp_tools_params:
            unique_tools[tool["function"]["name"]] = tool
        return list(unique_tools.values())

    def _process_think_response(self, response) -> bool:
        """Process think response and return whether to continue."""
        try:
            if self.tool_choices == ToolChoice.NONE:
                if response.tool_calls:
                    logger.warning(f"{self.name} selected tools but tool_choice is NONE")
                    return False
                if response.content:
                    asyncio.create_task(self.add_message("assistant", response.content))
                    return True
                return False

            asyncio.create_task(self.add_message("assistant", response.content, tool_calls=self.tool_calls))

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.content)
            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"{self.name} failed to think: {e}")
            asyncio.create_task(self.add_message("assistant", f"Error: {e}"))
            return False

    async def run(self, request: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """Execute agent with proper lifecycle management."""
        timeout = timeout or self._default_timeout

        # Acquire run lock
        try:
            async with asyncio.timeout(1.0):
                async with self._run_lock:
                    if self.state != AgentState.IDLE:
                        raise RuntimeError(f"Agent {self.name} is not in IDLE state")
                    self.state = AgentState.RUNNING
        except asyncio.TimeoutError:
            raise RuntimeError(f"Agent {self.name} is busy")

        if request is not None:
            await self.add_message("user", request)

        self._finish_reason_terminated = False
        self._final_response_content = None

        run_id = uuid.uuid4()
        runtime = self._create_runtime_context(run_id)
        results: List[str] = []

        try:
            # Before agent hooks
            if self._middleware_pipeline:
                self._middleware_pipeline.execute_before_agent(self._agent_state, runtime)

            # Plan phase
            if self.enable_plan_phase:
                await self._execute_plan_phase(runtime)

            # Main loop
            result = await self._run_main_loop(runtime, results)
            if result is not None:
                return result

            # Finish phase
            if self._middleware_pipeline:
                await self._execute_finish_phase(runtime, {"results": results})

            return "\n".join(results) if results else "No results"

        except Exception as e:
            logger.error(f"Error during agent run: {e}")
            raise
        finally:
            self._cleanup_run(runtime)

    async def _run_main_loop(self, runtime, results: List[str]) -> Optional[str]:
        """Execute main agent loop."""
        while self.current_step < self.max_steps and self.state == AgentState.RUNNING:
            self.current_step += 1
            runtime.current_step = self.current_step
            logger.info(f"Agent {self.name} step {self.current_step}/{self.max_steps}")

            # Calculate step timeout
            base_timeout = self.step_timeout or self._default_timeout
            step_timeout = self._calculate_timeout(base_timeout, for_step=True)

            try:
                step_result = await asyncio.wait_for(self.step(), timeout=step_timeout)
                if await self.is_stuck():
                    await self.handle_stuck_state()
            except asyncio.TimeoutError:
                logger.error(f"Step {self.current_step} timed out")
                break

            # Reflection
            if self.should_trigger_reflection():
                await self._execute_reflect_phase(runtime, {
                    "current_step": self.current_step,
                    "step_result": step_result,
                    "results": results,
                    "token_count": self.estimate_token_count(),
                })
                self._on_reflection_complete()

            # Check finish_reason termination
            if getattr(self, '_finish_reason_terminated', False):
                final = getattr(self, '_final_response_content', step_result)
                self._finish_reason_terminated = False
                if hasattr(self, '_final_response_content'):
                    delattr(self, '_final_response_content')
                return final

            results.append(f"Step {self.current_step}: {step_result}")
            logger.info(f"Step {self.current_step}: {step_result}")

        if self.current_step >= self.max_steps:
            results.append(f"Step {self.current_step}: Stuck in loop.")

        return None

    def _cleanup_run(self, runtime):
        """Cleanup after run completes."""
        if runtime and self._middleware_pipeline:
            try:
                runtime.messages = self.memory.get_messages() if hasattr(self.memory, 'get_messages') else []
                runtime.current_step = self.current_step
                self._middleware_pipeline.execute_after_agent(self._agent_state, runtime)
            except Exception as e:
                logger.error(f"Error in after_agent hooks: {e}")

        if self.state != AgentState.IDLE:
            logger.info(f"Resetting agent {self.name} to IDLE")
            self.state = AgentState.IDLE
            self.current_step = 0

    # ========================================================================
    # Step & Act
    # ========================================================================

    async def step(self) -> str:
        """Execute a single agent step."""
        should_act = await self.think()
        if not should_act:
            if self.state == AgentState.FINISHED:
                return "Task completed based on finish_reason signal"
            self.state = AgentState.FINISHED
            return "Thinking completed. No action needed."
        return await self.act()

    async def act(self) -> str:
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError("No tools to call")
            return self.memory.messages[-1].content or "No response"

        results = []
        for tool_call in self.tool_calls:
            try:
                result = await self.execute_tool(tool_call)
                logger.debug(f"Tool {tool_call.function.name} executed")
                if isinstance(result, str) and ("not healthy" in result.lower() or "execution failed" in result.lower()):
                    self.last_tool_error = result
            except Exception as e:
                result = f"Error executing tool {tool_call.function.name}: {str(e)}"
                logger.error(f"Tool execution failed: {e}")
                self.last_tool_error = str(e)

            await self.add_message("tool", result, tool_call_id=tool_call.id, tool_name=tool_call.function.name)
            results.append(result)
        return "\n\n".join(results)

    async def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute tool - routes through middleware if available."""
        kwargs = self._parse_tool_arguments(tool_call.function.arguments)

        if self._middleware_pipeline:
            return await self._execute_tool_with_middleware(tool_call, kwargs)
        return await self._execute_tool_direct(tool_call.function.name, kwargs)

    async def _execute_tool_with_middleware(self, tool_call: ToolCall, kwargs: dict) -> str:
        """Execute tool through middleware pipeline."""
        from spoon_ai.middleware.base import ToolCallRequest, ToolCallResult

        runtime = self._create_runtime_context() if hasattr(self, '_create_runtime_context') else None
        request = ToolCallRequest(
            tool_name=tool_call.function.name,
            arguments=kwargs,
            tool_call_id=tool_call.id,
            runtime=runtime,
            tool_call=tool_call
        )

        async def base_handler(req: ToolCallRequest) -> ToolCallResult:
            try:
                result = await self._execute_tool_direct(req.tool_name, req.arguments)
                return ToolCallResult.from_string(result)
            except Exception as e:
                return ToolCallResult.from_error(str(e))

        result = await self._middleware_pipeline.awrap_tool_call(request, base_handler)

        if result.error:
            self.last_tool_error = result.error
            raise Exception(result.error)
        return result.output

    async def _execute_tool_direct(self, tool_name: str, arguments: dict) -> str:
        """Direct tool execution without middleware."""
        if tool_name not in self.available_tools.tool_map:
            return f"Tool '{tool_name}' not found. Available: {list(self.available_tools.tool_map.keys())}"

        try:
            result = await self.available_tools.execute(name=tool_name, tool_input=arguments)
            observation = f"Observed output of cmd {tool_name} execution: {result}" if result else f"cmd {tool_name} executed"
            self._handle_special_tool(tool_name, result)
            return observation
        except Exception as e:
            print(f"❌ Tool execution error for {tool_name}: {e}")
            self.last_tool_error = str(e)
            raise

    @staticmethod
    def _parse_tool_arguments(arguments) -> dict:
        """Parse tool arguments from various formats."""
        if isinstance(arguments, str):
            arguments = arguments.strip()
            if not arguments:
                return {}
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {}
        elif isinstance(arguments, dict):
            return arguments
        return {}

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def consume_last_tool_error(self) -> Optional[str]:
        err = getattr(self, "last_tool_error", None)
        self.last_tool_error = None
        return err

    def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if self._is_special_tool(name) and self._should_finish_execution(name, result, **kwargs):
            self.state = AgentState.FINISHED

    def _is_special_tool(self, name: str) -> bool:
        return name.lower() in [n.lower() for n in self.special_tool_names]

    def _should_finish_execution(self, name: str, result: Any, **kwargs) -> bool:  # noqa: ARG002
        return True

    def _should_terminate_on_finish_reason(self, response) -> bool:
        """Check if agent should terminate based on finish_reason."""
        finish_reason = getattr(response, 'finish_reason', None)
        native_finish_reason = getattr(response, 'native_finish_reason', None)
        if finish_reason == "stop":
            return native_finish_reason in ["stop", "end_turn"]
        return False

    def clear(self):
        self.memory.clear()
        self.tool_calls = []
        self.state = AgentState.IDLE
        self.current_step = 0
        self._invalidate_mcp_cache()
        if hasattr(self, '_cache_lock'):
            delattr(self, '_cache_lock')
        logger.debug(f"🧹 {self.name} cleared")
