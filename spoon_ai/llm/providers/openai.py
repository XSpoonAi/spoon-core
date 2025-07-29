import os
from typing import List, Optional, Dict, Literal
from logging import getLogger

from openai import AsyncOpenAI

from spoon_ai.llm.base import LLMBase
from spoon_ai.llm.factory import LLMFactory
from spoon_ai.schema import Message, LLMConfig, LLMResponse, ToolCall, Function
from spoon_ai.utils.config_manager import ConfigManager

logger = getLogger(__name__)


class OpenAIConfig(LLMConfig):
    """OpenAI-specific configuration"""
    
    model: str = "gpt-4o"
    base_url: Optional[str] = None
    organization: Optional[str] = None


@LLMFactory.register("openai")
class OpenAIProvider(LLMBase):
    """OpenAI provider implementation using AsyncOpenAI client"""
    
    def __init__(self, config_path: str = "config.json", config_name: str = "llm"):
        # Use ConfigManager for all configuration (no TOML)
        self.config_manager = ConfigManager()
        
        # Load configuration using ConfigManager instead of TOML
        self.config = self._load_config_from_json()
        
        # Get API key with config.json -> environment fallback
        api_key = self.config_manager.get_api_key("openai") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config.json or OPENAI_API_KEY environment variable")
        
        # Get base_url from config.json or environment (for services like OpenRouter)
        base_url = self.config_manager.get("base_url") or os.getenv("BASE_URL")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=self.config.organization
        )
    
    def _load_config_from_json(self) -> OpenAIConfig:
        """Load configuration from config.json via ConfigManager"""
        # Get model from config.json or use default
        model_name = self.config_manager.get("model_name") or "gpt-4o"
        base_url = self.config_manager.get("base_url")
        
        return OpenAIConfig(
            model=model_name,
            base_url=base_url,
            max_tokens=4096,
            temperature=0.3
        )
        
    def _load_config(self, config_path: str, config_name: str) -> OpenAIConfig:
        """Load OpenAI-specific configuration (for compatibility only)"""
        # This method is for compatibility with LLMBase interface
        # We use _load_config_from_json() instead in the constructor
        return self._load_config_from_json()
    
    async def chat(
        self,
        messages: List[Message], 
        system_msgs: Optional[List[Message]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request to OpenAI"""
        # Format messages for OpenAI API
        formatted_messages = []
        
        # Add system messages first
        if system_msgs:
            for sys_msg in system_msgs:
                formatted_messages.append({
                    "role": "system",
                    "content": sys_msg.content
                })
        
        # Add user/assistant messages
        for message in messages:
            msg_dict = {"role": message.role}
            if message.content:
                msg_dict["content"] = message.content
            if message.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
            if message.name:
                msg_dict["name"] = message.name  
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id
            formatted_messages.append(msg_dict)
        
        try:
            response = await self.client.chat.completions.create(
                messages=formatted_messages,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content, text=content)
            
        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return LLMResponse(
                content=f"API request failed: {str(e)}", 
                text=f"API request failed: {str(e)}"
            )
    
    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Send text completion request to OpenAI"""
        # Create a user message
        message = Message(role="user", content=prompt)
        return await self.chat(messages=[message], **kwargs)
    
    async def chat_with_tools(
        self,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs
    ) -> LLMResponse:
        """Send chat request with tools to OpenAI"""
        # Format messages for OpenAI API
        formatted_messages = []
        
        # Add system messages first
        if system_msgs:
            for sys_msg in system_msgs:
                formatted_messages.append({
                    "role": "system", 
                    "content": sys_msg.content
                })
        
        # Add user/assistant messages
        for message in messages:
            msg_dict = {"role": message.role}
            if message.content:
                msg_dict["content"] = message.content
            if message.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
            if message.name:
                msg_dict["name"] = message.name
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id
            formatted_messages.append(msg_dict)
        
        try:
            response = await self.client.chat.completions.create(
                messages=formatted_messages,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs
            )
            
            # Extract message and finish_reason from OpenAI response
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            # Convert OpenAI tool calls to our ToolCall format
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tool_call.id,
                        type=tool_call.type,
                        function=Function(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments
                        )
                    ))
            
            # Map OpenAI finish reasons to standardized values
            standardized_finish_reason = finish_reason
            if finish_reason == "stop":
                standardized_finish_reason = "stop"
            elif finish_reason == "length":
                standardized_finish_reason = "length"
            elif finish_reason == "tool_calls":
                standardized_finish_reason = "tool_calls"
            elif finish_reason == "content_filter":
                standardized_finish_reason = "content_filter"
            
            content = message.content or ""
            
            return LLMResponse(
                content=content,
                text=content,
                tool_calls=tool_calls,
                finish_reason=standardized_finish_reason,
                native_finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return LLMResponse(
                content=f"API request failed: {str(e)}",
                text=f"API request failed: {str(e)}",
                tool_calls=[],
                finish_reason="error"
            )