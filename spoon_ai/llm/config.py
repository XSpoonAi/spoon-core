"""
Configuration management for LLM providers with validation and loading capabilities.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from logging import getLogger

from .errors import ConfigurationError

# Try to import toml, but make it optional
try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

logger = getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    name: str
    api_key: str
    base_url: Optional[str] = None
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 30
    retry_attempts: int = 3
    custom_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ConfigurationError("Provider name cannot be empty")
        if not self.api_key:
            raise ConfigurationError(f"API key is required for provider '{self.name}'")
        if self.max_tokens <= 0:
            raise ConfigurationError(f"max_tokens must be positive, got {self.max_tokens}")
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError(f"temperature must be between 0 and 2, got {self.temperature}")
        if self.timeout <= 0:
            raise ConfigurationError(f"timeout must be positive, got {self.timeout}")
        if self.retry_attempts < 0:
            raise ConfigurationError(f"retry_attempts must be non-negative, got {self.retry_attempts}")
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            'name': self.name,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'custom_headers': self.custom_headers.copy(),
            'extra_params': self.extra_params.copy()
        }


class ConfigurationManager:
    """Manages configuration loading and validation for LLM providers."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load .env file if available
        self._load_dotenv()
        
        self.config_path = config_path or self._find_config_file()
        self._config_cache: Dict[str, Any] = {}
        self._provider_configs: Dict[str, ProviderConfig] = {}
        self._load_config()
    
    def _load_dotenv(self) -> None:
        """Load environment variables from .env file if available."""
        if HAS_DOTENV:
            # Try to load .env from current directory and parent directories
            env_paths = ['.env', '../.env', '../../.env']
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment variables from {env_path}")
                    break
            else:
                # Try to load from default location
                load_dotenv()
        else:
            logger.debug("python-dotenv not available, skipping .env file loading")
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in common locations.
        
        Returns:
            Optional[str]: Path to configuration file if found
        """
        possible_paths = [
            "config.json",
            "config/config.json", 
            "config.toml",
            "config/config.toml",
            os.path.expanduser("~/.spoon/config.json"),
            os.path.expanduser("~/.spoon/config.toml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found configuration file: {path}")
                return path
        
        logger.warning("No configuration file found, using environment variables only")
        return None
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_path or not os.path.exists(self.config_path):
            logger.info("No configuration file available, using environment variables")
            self._config_cache = {}
            return
        
        try:
            if self.config_path.endswith('.json'):
                with open(self.config_path, 'r') as f:
                    self._config_cache = json.load(f)
            elif self.config_path.endswith('.toml'):
                if not HAS_TOML:
                    raise ConfigurationError(
                        "TOML configuration file found but 'toml' package not installed. "
                        "Please install it with: pip install toml"
                    )
                self._config_cache = toml.load(self.config_path)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {self.config_path}")
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {self.config_path}: {str(e)}",
                context={"config_path": self.config_path, "error": str(e)}
            )
    
    def load_provider_config(self, provider_name: str) -> ProviderConfig:
        """Load and validate provider configuration.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            ProviderConfig: Validated provider configuration
            
        Raises:
            ConfigurationError: If configuration is invalid or missing
        """
        if provider_name in self._provider_configs:
            return self._provider_configs[provider_name]
        
        # Get provider-specific configuration
        provider_config = self._get_provider_config_dict(provider_name)
        
        try:
            config = ProviderConfig(
                name=provider_name,
                api_key=provider_config.get('api_key', ''),
                base_url=provider_config.get('base_url'),
                model=provider_config.get('model', ''),
                max_tokens=provider_config.get('max_tokens', 4096),
                temperature=provider_config.get('temperature', 0.3),
                timeout=provider_config.get('timeout', 30),
                retry_attempts=provider_config.get('retry_attempts', 3),
                custom_headers=provider_config.get('custom_headers', {}),
                extra_params=provider_config.get('extra_params', {})
            )
            
            # Cache the validated config
            self._provider_configs[provider_name] = config
            return config
            
        except Exception as e:
            raise ConfigurationError(
                f"Invalid configuration for provider '{provider_name}': {str(e)}",
                config_key=provider_name,
                context={"provider_config": provider_config, "error": str(e)}
            )
    
    def _get_provider_config_dict(self, provider_name: str) -> Dict[str, Any]:
        """Get provider configuration dictionary from various sources.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Dict[str, Any]: Provider configuration dictionary
        """
        config = {}
        
        # 1. Load from configuration file
        if self._config_cache:
            # Try provider-specific section first
            if 'providers' in self._config_cache and provider_name in self._config_cache['providers']:
                config.update(self._config_cache['providers'][provider_name])
            
            # Try legacy api_keys section
            if 'api_keys' in self._config_cache and provider_name in self._config_cache['api_keys']:
                config['api_key'] = self._config_cache['api_keys'][provider_name]
            
            # Try global LLM config
            if 'llm' in self._config_cache:
                llm_config = self._config_cache['llm']
                if llm_config.get('provider') == provider_name:
                    config.update(llm_config)
        
        # 2. Load from environment variables
        env_mappings = {
            'api_key': f'{provider_name.upper()}_API_KEY',
            'base_url': f'{provider_name.upper()}_BASE_URL',
            'model': f'{provider_name.upper()}_MODEL',
            'max_tokens': f'{provider_name.upper()}_MAX_TOKENS',
            'temperature': f'{provider_name.upper()}_TEMPERATURE',
            'timeout': f'{provider_name.upper()}_TIMEOUT'
        }
        
        for config_key, env_key in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                # Convert string values to appropriate types
                if config_key in ['max_tokens', 'timeout']:
                    try:
                        config[config_key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_key}: {env_value}")
                elif config_key == 'temperature':
                    try:
                        config[config_key] = float(env_value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_key}: {env_value}")
                else:
                    config[config_key] = env_value
        
        # 2.1. Fallback to generic environment variables for backward compatibility
        if 'base_url' not in config:
            generic_base_url = os.getenv('BASE_URL')
            if generic_base_url:
                config['base_url'] = generic_base_url
                logger.info(f"Using generic BASE_URL for {provider_name}: {generic_base_url}")
        
        # 3. Apply provider-specific defaults
        defaults = self._get_provider_defaults(provider_name)
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _get_provider_defaults(self, provider_name: str) -> Dict[str, Any]:
        """Get default configuration for a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Dict[str, Any]: Default configuration
        """
        defaults = {
            'openai': {
                'model': 'gpt-4.1',
                'base_url': 'https://api.openai.com/v1'
            },
            'openrouter': {
                'model': 'openai/gpt-4.1',
                'base_url': 'https://openrouter.ai/api/v1'
            },
            'deepseek': {
                'model': 'deepseek-chat',
                'base_url': 'https://api.deepseek.com/v1'
            },
            'anthropic': {
                'model': 'claude-sonnet-4-20250514',
                'base_url': 'https://api.anthropic.com'
            },
            'gemini': {
                'model': 'gemini-2.5-pro',
                'base_url': 'https://generativelanguage.googleapis.com'
            }
        }
        
        return defaults.get(provider_name, {})
    
    def validate_config(self, config: ProviderConfig) -> bool:
        """Validate provider configuration.
        
        Args:
            config: Provider configuration to validate
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # ProviderConfig.__post_init__ handles validation
            # If we get here without exception, config is valid
            return True
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def get_default_provider(self) -> str:
        """Get default provider from configuration with intelligent selection.
        
        Returns:
            str: Default provider name
        """
        # 1. Check explicit configuration file setting
        if self._config_cache:
            # Check llm_settings.default_provider first (new format)
            if 'llm_settings' in self._config_cache:
                provider = self._config_cache['llm_settings'].get('default_provider')
                if provider:
                    logger.info(f"Using provider from config file llm_settings: {provider}")
                    return provider
            
            # Check legacy llm.provider format
            if 'llm' in self._config_cache:
                provider = self._config_cache['llm'].get('provider')
                if provider:
                    logger.info(f"Using provider from config file llm: {provider}")
                    return provider
        
        # 2. Check environment variable for explicit preference
        env_provider = os.getenv('DEFAULT_LLM_PROVIDER')
        if env_provider:
            logger.info(f"Using provider from DEFAULT_LLM_PROVIDER: {env_provider}")
            return env_provider
        
        # 3. Intelligent selection based on available API keys and quality
        # Priority order: anthropic (best quality) -> openai -> openrouter -> deepseek -> gemini
        provider_priority = ['anthropic', 'openai', 'openrouter', 'deepseek', 'gemini']
        available_providers = []
        
        for provider in provider_priority:
            try:
                config = self._get_provider_config_dict(provider)
                if config.get('api_key'):
                    available_providers.append(provider)
                    logger.debug(f"Found API key for {provider}")
            except Exception as e:
                logger.debug(f"No valid config for {provider}: {e}")
                continue
        
        if available_providers:
            selected = available_providers[0]  # First in priority order
            logger.info(f"Intelligently selected {selected} as default provider (available: {available_providers})")
            return selected
        
        # 4. Ultimate fallback
        logger.warning("No API keys found for any provider, falling back to openai")
        return 'openai'
    
    def get_fallback_chain(self) -> List[str]:
        """Get fallback chain from configuration.
        
        Returns:
            List[str]: List of provider names in fallback order
        """
        # Check llm_settings.fallback_chain in config file
        if self._config_cache and 'llm_settings' in self._config_cache:
            fallback_chain = self._config_cache['llm_settings'].get('fallback_chain')
            if fallback_chain:
                logger.info(f"Using fallback chain from config file: {fallback_chain}")
                return fallback_chain
        
        # Fallback to intelligent selection based on available providers
        available_providers = self.get_available_providers_by_priority()
        if available_providers:
            logger.info(f"Using intelligent fallback chain: {available_providers}")
            return available_providers
        
        # Ultimate fallback
        logger.warning("No fallback chain configured, using default")
        return ['openai']
    
    def list_configured_providers(self) -> List[str]:
        """List all configured providers.
        
        Returns:
            List[str]: List of provider names that have configuration
        """
        providers = set()
        
        # From configuration file
        if self._config_cache:
            if 'providers' in self._config_cache:
                providers.update(self._config_cache['providers'].keys())
            if 'api_keys' in self._config_cache:
                providers.update(self._config_cache['api_keys'].keys())
        
        # From environment variables
        for provider in ['openai', 'openrouter', 'deepseek', 'anthropic', 'gemini']:
            if os.getenv(f'{provider.upper()}_API_KEY'):
                providers.add(provider)
        
        return list(providers)
    
    def get_available_providers_by_priority(self) -> List[str]:
        """Get available providers ordered by priority and quality.
        
        Returns:
            List[str]: List of available provider names in priority order
        """
        # Define priority order based on quality and capabilities
        priority_order = ['anthropic', 'openai', 'openrouter', 'deepseek', 'gemini']
        available_providers = []
        
        for provider in priority_order:
            try:
                config = self._get_provider_config_dict(provider)
                if config.get('api_key'):
                    available_providers.append(provider)
            except Exception:
                continue
        
        return available_providers
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all providers and their availability.
        
        Returns:
            Dict[str, Dict[str, Any]]: Provider information including availability
        """
        provider_info = {}
        
        for provider in ['anthropic', 'openai', 'openrouter', 'deepseek', 'gemini']:
            try:
                config = self._get_provider_config_dict(provider)
                has_api_key = bool(config.get('api_key'))
                
                provider_info[provider] = {
                    'available': has_api_key,
                    'model': config.get('model', 'Unknown'),
                    'base_url': config.get('base_url'),
                    'configured_via': 'environment' if os.getenv(f'{provider.upper()}_API_KEY') else 'config_file'
                }
            except Exception as e:
                provider_info[provider] = {
                    'available': False,
                    'error': str(e)
                }
        
        return provider_info
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config_cache.clear()
        self._provider_configs.clear()
        self._load_config()
        logger.info("Configuration reloaded")