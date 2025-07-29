import os
from typing import Dict, Optional, Type

import toml

from spoon_ai.llm.base import LLMBase
from spoon_ai.schema import LLMConfig
from logging import getLogger
logger = getLogger(__name__)


class LLMFactory:
    """LLM factory class, used to create different LLM instances"""
    
    _providers: Dict[str, Type[LLMBase]] = {}
    
    @classmethod
    def register(cls, provider_name: str):
        """Register LLM provider
        
        Args:
            provider_name: Provider name
            
        Returns:
            Decorator function
        """
        def decorator(provider_class: Type[LLMBase]):
            cls._providers[provider_name] = provider_class
            return provider_class
        return decorator
    
    @classmethod
    def create(
        cls, 
        provider: Optional[str] = None, 
        config_path: str = "config/config.toml", 
        config_name: str = "llm"
    ) -> LLMBase:
        """Create LLM instance
        
        Args:
            provider: Provider name, if None, read from configuration file
            config_path: Configuration file path
            config_name: Configuration name
            
        Returns:
            LLMBase: LLM instance
            
        Raises:
            ValueError: If provider does not exist
        """
        # If provider is not specified, read from configuration file
        if provider is None:
            provider = cls._get_provider_from_config(config_path, config_name)
        
        # Check if provider exists
        if provider not in cls._providers:
            available_providers = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Provider {provider} does not exist, available providers: {available_providers}"
            )
        
        # Create LLM instance
        provider_class = cls._providers[provider]
        return provider_class(config_path=config_path, config_name=config_name)
    
    @staticmethod
    def _get_provider_from_config(config_path: str, config_name: str) -> str:
        """Read provider name from configuration file
        
        Args:
            config_path: Configuration file path
            config_name: Configuration name
            
        Returns:
            str: Provider name
        """
        # Default to OpenAI
        default_provider = "openai"
        
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file {config_path} does not exist, using default provider {default_provider}")
            return default_provider
            
        try:
            config_data = toml.load(config_path)
            llm_config = config_data.get(config_name, {})
            provider = llm_config.get("provider", default_provider)
            return provider
        except Exception as e:
            logger.error(f"Failed to read configuration: {e}")
            return default_provider 