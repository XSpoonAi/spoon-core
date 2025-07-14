import json
import os
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
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                "api_keys": {},
                "base_url": "",
                "default_agent": "default"
            }

        # Create default configuration if loading fails or file doesn't exist
        default_config = {
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
                "deepseek": os.environ.get("DEEPSEEK_API_KEY", "")
            },
            "base_url": os.environ.get("BASE_URL", ""),
            "default_agent": "default"
        }
        self._save_config(default_config)
        return default_config

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
        """Get API key for specified provider"""
        return self.get(f"api_keys.{provider}")

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
        """Get LLM provider from model name or configuration"""
        if self.get("llm_provider"):
            return self.get("llm_provider")
        else:
            return 'openai'
