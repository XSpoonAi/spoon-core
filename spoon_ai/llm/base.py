import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import toml
from pydantic import BaseModel, Field

from logging import getLogger

from spoon_ai.schema import Message, LLMConfig, LLMResponse
logger = getLogger(__name__)



class LLMBase(ABC):
    """Base abstract class for LLM, defining interfaces that all LLM providers must implement"""
    
    def __init__(self, config_path: str = "config/config.toml", config_name: str = "llm"):
        """Initialize LLM interface
        
        Args:
            config_path: Configuration file path
            config_name: Configuration name
        """
        self.config = self._load_config(config_path, config_name)
        self.output_queue = asyncio.Queue()
        self.task_done = asyncio.Event()
        
    def _load_config(self, config_path: str, config_name: str) -> LLMConfig:
        """Load configuration
        
        Args:
            config_path: Configuration file path
            config_name: Configuration name
            
        Returns:
            LLMConfig: LLM configuration
        """
        if not os.path.exists(config_path):
            # Use default configuration
            logger.warning(f"Configuration file {config_path} does not exist, using default configuration")
            return LLMConfig()
            
        try:
            config_data = toml.load(config_path)
            llm_config = config_data.get(config_name, {})
            return LLMConfig(**llm_config)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return LLMConfig()
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request to LLM and get response
        
        Args:
            messages: List of messages
            system_msgs: List of system messages
            **kwargs: Other parameters
            
        Returns:
            LLMResponse: LLM response
        """
        pass
    
    @abstractmethod
    async def completion(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Send text completion request to LLM and get response
        
        Args:
            prompt: Prompt text
            **kwargs: Other parameters
            
        Returns:
            LLMResponse: LLM response
        """
        pass
    
    @abstractmethod
    async def chat_with_tools(
        self,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs
    ) -> LLMResponse:
        """Send chat request that may contain tool calls to LLM and get response
        
        Args:
            messages: List of messages
            system_msgs: List of system messages
            tools: List of tools
            tool_choice: Tool selection mode
            **kwargs: Other parameters
            
        Returns:
            LLMResponse: LLM response
        """
        pass
    
    async def generate_image(
        self,
        prompt: str,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate image (optional implementation)
        
        Args:
            prompt: Prompt text
            **kwargs: Other parameters
            
        Returns:
            Union[str, List[str]]: Image URL or list of URLs
        """
        logger.warning(f"Current LLM provider {self.__class__.__name__} does not support image generation")
        return "" 

    def reset_output_handler(self):
        """Reset output handler"""
        self.output_queue = asyncio.Queue()
        self.task_done = asyncio.Event()