import json
import asyncio
from logging import getLogger
from typing import Any, List
import logging

from pydantic import Field
from termcolor import colored

from spoon_ai.agents.react import ReActAgent
from spoon_ai.prompts.toolcall import \
    NEXT_STEP_PROMPT as TOOLCALL_NEXT_STEP_PROMPT
from spoon_ai.prompts.toolcall import SYSTEM_PROMPT as TOOLCALL_SYSTEM_PROMPT
from spoon_ai.schema import TOOL_CHOICE_TYPE, AgentState, ToolCall, ToolChoice
from spoon_ai.tools import Terminate, ToolManager
from mcp.types import Tool as MCPTool
from spoon_ai.tools.mcp_tool import MCPTool as SpoonMCPTool

logging.getLogger("spoon_ai").setLevel(logging.INFO)

logger = getLogger("spoon_ai")

class ToolCallAgent(ReActAgent):
    
    name: str = "toolcall"
    description: str = "Useful when you need to call a tool"
    
    system_prompt: str = TOOLCALL_SYSTEM_PROMPT
    next_step_prompt: str = TOOLCALL_NEXT_STEP_PROMPT
    
    avaliable_tools: ToolManager = Field(default_factory=ToolManager)
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])
    
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO # type: ignore
    
    tool_calls: List[ToolCall] = Field(default_factory=list)
    
    output_queue: asyncio.Queue = Field(default_factory=asyncio.Queue)
    
    async def think(self) -> bool:
        if self.next_step_prompt:
            self.add_message("user", self.next_step_prompt)
        
        mcp_tools = []
        if hasattr(self, "list_mcp_tools"):
            mcp_tools = await self.list_mcp_tools()

        def convert_mcp_tool(tool: MCPTool) -> SpoonMCPTool:
            return SpoonMCPTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            ).to_param()
        
        all_tools = self.avaliable_tools.to_params()
        mcp_tools_params = [convert_mcp_tool(tool) for tool in mcp_tools]
        unique_tools = {}
        for tool in all_tools + mcp_tools_params:
            tool_name = tool["function"]["name"]
            unique_tools[tool_name] = tool
        unique_tools_list = list(unique_tools.values())

        response = await self.llm.ask_tool(
            messages=self.memory.messages,
            system_msg=self.system_prompt,
            tools=unique_tools_list,
            tool_choice=self.tool_choices,
            output_queue=self.output_queue,
        )

        self.tool_calls = response.tool_calls
        
        logger.info(colored(f"🤔 {self.name}'s thoughts: {response.content}", "cyan"))
        tool_count = len(self.tool_calls) if self.tool_calls else 0
        logger.info(colored(f"🛠️ {self.name} selected {tool_count} tools: {self.tool_calls}", "green" if tool_count else "yellow"))
        
        if self.output_queue:
            self.output_queue.put_nowait({"content": response.content})
            self.output_queue.put_nowait({"tool_calls": response.tool_calls})

        try:
            if self.tool_choices == ToolChoice.NONE:
                if response.tool_calls:
                    logger.warning(f"{self.name} selected {len(self.tool_calls)} tools, but tool_choice is NONE")
                    return False
                if response.content:
                    self.add_message("assistant", response.content)
                    return True
                return False
            self.add_message("assistant", response.content, tool_calls=self.tool_calls)
            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.content)
            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"{self.name} failed to think: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.add_message("assistant", f"Error encountered while thinking: {e}")
            return False
        
    async def act(self) -> str:
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError("No tools to call")
            return self.memory.messages[-1].content or "No response from assistant"
        
        results = []
        for tool_call in self.tool_calls:
            result = await self.execute_tool(tool_call)
            logger.info(f"Tool {tool_call.function.name} executed with result: {result}")
            self.add_message("tool", result, tool_call_id=tool_call.id)
            results.append(result)
        return "\n\n".join(results)
        
    async def execute_tool(self, tool_call: ToolCall) -> str:
        if tool_call.function.name not in self.avaliable_tools.tool_map:
            if not hasattr(self, "call_mcp_tool"):
                raise ValueError(f"Tool {tool_call.function.name} not found")
            kwargs = json.loads(tool_call.function.arguments or {})
            result = await self.call_mcp_tool(tool_call.function.name, **kwargs)
            return result
        if not tool_call or not tool_call.function or not tool_call.function.name:
            return "Error: Invalid tool call"
        
        name = tool_call.function.name
        if name not in self.avaliable_tools.tool_map:
            return f"Error: Tool {name} not found"
        
        try:
            args = json.loads(tool_call.function.arguments or {})
            result = await self.avaliable_tools.execute(name=name, tool_input=args)
            
            observation = (
                f"Observed output of cmd {name} execution: {result}"
                if result
                else f"cmd {name} execution without any output"
            )
            
            self._handle_special_tool(name, result)
            return observation
        
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {e}"
            
        
    def _handle_special_tool(self, name: str, result:Any, **kwargs):
        if not self._is_special_tool(name):
            return
        if self._should_finish_execution(name, result, **kwargs):
            self.state = AgentState.FINISHED
        return
    
    def _is_special_tool(self, name: str) -> bool:
        return name.lower() in [n.lower() for n in self.special_tool_names]
    
    def _should_finish_execution(self, name: str, result: Any, **kwargs) -> bool:
        return True
    
    def reset_state(self):
        """Reset only the agent state and step counter without clearing memory"""
        self.tool_calls = []
        self.state = AgentState.IDLE
        self.current_step = 0
    
    def clear(self):
        self.memory.clear()
        self.reset_state()