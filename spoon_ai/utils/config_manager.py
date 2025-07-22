import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration management class for user settings like API keys"""

    def __init__(self):
        """Initialize the configuration manager"""
        # Use relative path from current working directory
        self.config_file = Path("config.json")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        # Try to load existing configuration
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default configuration if file doesn't exist
                return {
                    "api_keys": {},
                    "base_url": "",
                    "default_agent": "react"
                }
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                "api_keys": {},
                "base_url": "",
                "default_agent": "react"
            }

    def _is_placeholder_value(self, value: str) -> bool:
        """
        Check if the given value is a placeholder API key.

        This method identifies common placeholder patterns used in configuration
        templates and documentation examples. It helps distinguish between actual
        API keys and placeholder values that should be replaced.

        Args:
            value (str): The API key value to check

        Returns:
            bool: True if the value appears to be a placeholder, False otherwise

        Placeholder patterns detected:
        - Empty strings or None values
        - OpenAI placeholders: "sk-your-openai-api-key-here", "sk-your-*-key-here"
        - Anthropic placeholders: "sk-ant-your-anthropic-api-key-here", "sk-ant-your-*-key-here"
        - DeepSeek placeholders: "your-deepseek-api-key-here", "your-*-api-key-here"
        - Generic placeholders: "your-api-key-here", "insert-your-key-here"
        - Template patterns: "YOUR_API_KEY", "API_KEY_HERE"
        """
        if not value or not isinstance(value, str):
            return True

        # Convert to lowercase for case-insensitive matching
        value_lower = value.lower().strip()

        # Check for empty or whitespace-only strings
        if not value_lower:
            return True

        # Define placeholder patterns using regex
        placeholder_patterns = [
            # OpenAI placeholder patterns
            r'^sk-your-.*-key-here$',
            r'^sk-your-openai-api-key-here$',

            # Anthropic placeholder patterns
            r'^sk-ant-your-.*-key-here$',
            r'^sk-ant-your-anthropic-api-key-here$',

            # DeepSeek and generic placeholder patterns
            r'^your-.*-api-key-here$',
            r'^your-deepseek-api-key-here$',
            r'^your-api-key-here$',

            # Generic template patterns
            r'^insert-your-.*-key.*$',
            r'^your_api_key$',
            r'^api_key_here$',
            r'^your-.*-key$',

            # Common placeholder text
            r'^(api_key|your_key|insert_key|placeholder|example_key)$',
            r'^<.*>$',  # Angle bracket placeholders like <YOUR_API_KEY>
            r'^\[.*\]$',  # Square bracket placeholders like [YOUR_API_KEY]
            r'^\{.*\}$',  # Curly brace placeholders like {YOUR_API_KEY}
        ]

        # Check if value matches any placeholder pattern
        for pattern in placeholder_patterns:
            if re.match(pattern, value_lower):
                return True

        # Additional checks for obvious placeholder indicators
        placeholder_keywords = [
            'placeholder', 'example', 'sample', 'demo', 'test',
            'your-key', 'insert', 'replace', 'change-me'
        ]

        for keyword in placeholder_keywords:
            if keyword in value_lower:
                return True

        return False

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration item"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration item"""
        keys = key.split('.')
        config = self.config
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)

    def list_config(self) -> Dict[str, Any]:
        """List all configuration items"""
        return self.config

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for specified provider with config-first priority.

        Priority order:
        1. Config file value (if exists and not a placeholder)
        2. Environment variable fallback

        Args:
            provider (str): The API provider name (e.g., 'openai', 'anthropic', 'deepseek')

        Returns:
            Optional[str]: The API key if found, None otherwise
        """
        # First, try to get the value from config file
        config_value = self.get(f"api_keys.{provider}")

        # If config value exists and is not a placeholder, use it
        if config_value and not self._is_placeholder_value(config_value):
            return config_value

        # Otherwise, fall back to environment variable
        env_key = f"{provider.upper()}_API_KEY"
        env_value = os.environ.get(env_key)

        return env_value

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for specified provider"""
        self.set(f"api_keys.{provider}", api_key)
        # Also set environment variable
        os.environ[f"{provider.upper()}_API_KEY"] = api_key

    def get_model_name(self) -> Optional[str]:
        """Get configured model name"""
        return self.get("model_name")

    def get_base_url(self) -> Optional[str]:
        """Get configured base URL"""
        return self.get("base_url")

    def get_llm_provider(self) -> Optional[str]:
        """
        Get LLM provider from configuration with intelligent detection.

        Priority:
        1. Explicit llm_provider field in config
        2. Auto-detect from configured API keys (config-first priority)
        3. Return None to let ChatBot handle provider selection

        Returns:
            Optional[str]: Provider name or None for auto-detection
        """
        # First check for explicit llm_provider setting
        explicit_provider = self.get("llm_provider")
        if explicit_provider:
            return explicit_provider

        # Auto-detect from configured API keys (config-first priority)
        api_keys = self.get("api_keys", {})

        # Check for valid (non-placeholder) API keys in priority order
        for provider in ["anthropic", "openai", "deepseek"]:
            if provider in api_keys:
                key_value = api_keys[provider]
                if key_value and not self._is_placeholder_value(key_value):
                    return provider

        # If no valid API keys found in config, return None to let ChatBot handle it
        return None
