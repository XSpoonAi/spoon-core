"""
Configuration management for LLM providers using environment variables.
"""

import os
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from logging import getLogger

from .errors import ConfigurationError

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

logger = getLogger(__name__)

PLACEHOLDER_LITERALS = {
    "your_api_key",
    "api_key_here",
    "insert_api_key_here",
    "your_base_url_here",
    "your-base-url-here",
}

PLACEHOLDER_REGEXES = [
    re.compile(r"^your[-_a-z0-9]*here$"),
    re.compile(r"^sk[-_a-z0-9]*your[-_a-z0-9]*$"),
]


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
    """Manages environment-driven configuration for LLM providers."""

    def __init__(self) -> None:
        """Initialize configuration manager and load environment variables."""
        self._load_dotenv()
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

    def _load_config(self) -> None:
        """Initialize in-memory configuration cache (environment only)."""
        self._config_cache = {}

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
                timeout=provider_config.get('timeout', 300),
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
        """Get provider configuration dictionary with flexible fallback logic.

        Configuration Priority (highest to lowest):
        1. providers[provider_name] section (complete provider config)
        2. api_keys[provider_name] (for API key only, if not in providers)
        3. Environment variables (provider-specific)
        4. Default values (provider-specific defaults)

        This allows flexible configuration:
        - Only api_keys section: providers section not needed
        - Only providers section: api_keys section not needed
        - Mixed: providers can omit api_key if in api_keys section
        - Minimal config: only api_key needed, rest uses defaults

        Args:
            provider_name: Name of the provider

        Returns:
            Dict[str, Any]: Provider configuration dictionary with defaults
        """
        config = {}

        # 1. Load from configuration file with smart fallback
        if self._config_cache:
            # Try provider-specific section first (highest priority)
            if 'providers' in self._config_cache and provider_name in self._config_cache['providers']:
                provider_config = self._config_cache['providers'][provider_name]
                config.update(provider_config)
                logger.debug(f"Loaded provider config from providers section for {provider_name}")

            # Smart API key fallback: use api_keys section if no api_key in providers
            if not config.get('api_key') and 'api_keys' in self._config_cache:
                if provider_name in self._config_cache['api_keys']:
                    api_key = self._config_cache['api_keys'][provider_name]
                    if api_key and api_key.strip():  # Only use non-empty api_keys
                        config['api_key'] = api_key
                        logger.debug(f"Using API key from api_keys section for {provider_name}")

            # Try global LLM config (lowest priority from injected configuration)
            if 'llm' in self._config_cache:
                llm_config = self._config_cache['llm']
                if llm_config.get('provider') == provider_name:
                    # Only update if not already set (preserve higher priority values)
                    for key, value in llm_config.items():
                        if key not in config or not config[key]:
                            config[key] = value

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
            if not env_value:
                continue

            if config_key in ['api_key', 'base_url', 'model'] and self._is_placeholder_value(env_value):
                message = f"Ignoring placeholder value provided via {env_key}"
                if config_key == 'base_url':
                    raise ConfigurationError(
                        provider_name,
                        f"{env_key} is set to a placeholder value. Please provide a valid URL."
                    )
                logger.debug(message)
                continue

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
                config[config_key] = env_value.strip()

        # 2.1. Fallback to generic environment variables for backward compatibility
        if 'base_url' not in config:
            generic_base_url = os.getenv('BASE_URL')
            if generic_base_url:
                if self._is_placeholder_value(generic_base_url):
                    raise ConfigurationError(
                        provider_name,
                        "BASE_URL is set to the placeholder value 'your_base_url_here'. "
                        "Remove it or replace it with a valid HTTPS endpoint."
                    )
                config['base_url'] = generic_base_url.strip()
                logger.info(f"Using generic BASE_URL for {provider_name}: {config['base_url']}")

        # 3. Apply provider-specific defaults
        defaults = self._get_provider_defaults(provider_name)
        for key, value in defaults.items():
            if key not in config:
                config[key] = value

        return config

    @staticmethod
    def _is_placeholder_value(value: Optional[str]) -> bool:
        """Detect whether a configuration value still uses template placeholders."""
        if value is None:
            return True

        normalized = value.strip().lower()
        if not normalized:
            return True

        if normalized in PLACEHOLDER_LITERALS:
            return True

        for regex in PLACEHOLDER_REGEXES:
            if regex.match(normalized):
                return True

        # Common delimiters for documented placeholders
        if (normalized.startswith("<") and normalized.endswith(">")) or \
           (normalized.startswith("[") and normalized.endswith("]")) or \
           (normalized.startswith("{") and normalized.endswith("}")):
            return True

        return False

    def _get_provider_defaults(self, provider_name: str) -> Dict[str, Any]:
        """Get comprehensive default configuration for a provider.

        Provides sensible defaults for all configuration options, allowing
        minimal configuration where only api_key is required.

        Args:
            provider_name: Name of the provider

        Returns:
            Dict[str, Any]: Complete default configuration
        """
        # Common defaults for all providers
        common_defaults = {
            'temperature': 0.3,
            'timeout': 30,
            'retry_attempts': 3,
            'custom_headers': {},
            'extra_params': {}
        }

        # Provider-specific defaults
        provider_defaults = {
            'openai': {
                'model': 'gpt-4.1',
                'base_url': 'https://api.openai.com/v1',
                **common_defaults
            },
            'openrouter': {
                'model': 'anthropic/claude-sonnet-4',
                'base_url': 'https://openrouter.ai/api/v1',
                'custom_headers': {
                    'HTTP-Referer': 'https://github.com/spoon-ai/spoon-core',
                    'X-Title': 'SpoonAI Agent'
                },
                **common_defaults
            },
            'deepseek': {
                'model': 'deepseek-reasoner',
                'base_url': 'https://api.deepseek.com/v1',
                'max_tokens': 65536,  # DeepSeek supports larger context
                'temperature': 0.2,   # Slightly lower for reasoning model
                **{k: v for k, v in common_defaults.items() if k not in ['max_tokens', 'temperature']}
            },
            'anthropic': {
                'model': 'claude-sonnet-4-20250514',
                'base_url': 'https://api.anthropic.com',
                'max_tokens': 63000,  # Claude Sonnet-4 supports max_token of context <64000
                'temperature': 0.1,   # Lower temperature for Claude
                **{k: v for k, v in common_defaults.items() if k != 'temperature'}
            },
            'gemini': {
                'model': 'gemini-2.5-pro',
                'max_tokens': 20000,
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'temperature': 0.1,   # Lower temperature for Gemini
                **{k: v for k, v in common_defaults.items() if k != 'temperature'}
            },
            'ollama': {
                # Local Ollama server (no real API key required; placeholder keeps config validation consistent)
                'api_key': 'ollama',
                'model': 'llama3.2',
                'base_url': 'http://localhost:11434',
                **common_defaults
            }
        }

        # Return provider-specific defaults or common defaults for unknown providers
        return provider_defaults.get(provider_name, common_defaults)

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
        # 1. Check injected configuration data first
        if self._config_cache:
            # Check llm_settings.default_provider first (new format)
            if 'llm_settings' in self._config_cache:
                provider = self._config_cache['llm_settings'].get('default_provider')
                if provider:
                    logger.info(f"Using provider from injected configuration llm_settings: {provider}")
                    return provider

            # Check legacy llm.provider format
            if 'llm' in self._config_cache:
                provider = self._config_cache['llm'].get('provider')
                if provider:
                    logger.info(f"Using provider from injected configuration llm: {provider}")
                    return provider

        # 2. Check environment variable for explicit preference
        env_provider = os.getenv("LLM_PROVIDER") or os.getenv("DEFAULT_LLM_PROVIDER")
        if env_provider:
            normalized = env_provider.strip().lower()
            if normalized:
                logger.info(
                    "Using provider from environment (LLM_PROVIDER/DEFAULT_LLM_PROVIDER): %s",
                    normalized,
                )
                return normalized

        # 3. Intelligent selection based on available API keys and quality
        # Priority order: openai (GPT-4.1 default) -> anthropic -> openrouter -> deepseek -> gemini
        provider_priority = ['openai', 'anthropic', 'openrouter', 'deepseek', 'gemini']
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
        def _dedupe_preserve_order(items: List[str]) -> List[str]:
            seen = set()
            result: List[str] = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                result.append(item)
            return result

        def _filter_available_providers(items: List[str]) -> List[str]:
            usable: List[str] = []
            for provider in items:
                try:
                    config = self._get_provider_config_dict(provider)
                except Exception:
                    # Invalid config (e.g. placeholder BASE_URL). Treat as unavailable.
                    continue

                api_key = config.get("api_key")
                if not api_key:
                    continue
                if isinstance(api_key, str) and self._is_placeholder_value(api_key):
                    continue
                usable.append(provider)
            return usable

        # Check environment variable override first
        env_chain = os.getenv("LLM_FALLBACK_CHAIN")
        if env_chain:
            raw = [provider.strip().lower() for provider in env_chain.split(",") if provider.strip()]
            chain = _dedupe_preserve_order(raw)
            filtered = _filter_available_providers(chain)
            if filtered:
                if filtered != chain:
                    logger.info(
                        "Filtered fallback chain to configured providers. requested=%s usable=%s",
                        chain,
                        filtered,
                    )
                else:
                    logger.info("Using fallback chain from environment: %s", filtered)
                return filtered

        # Support programmatic cache injections (used by higher-level tooling)
        if self._config_cache and 'llm_settings' in self._config_cache:
            fallback_chain = self._config_cache['llm_settings'].get('fallback_chain')
            if isinstance(fallback_chain, list) and fallback_chain:
                raw = [str(provider).strip().lower() for provider in fallback_chain if str(provider).strip()]
                chain = _dedupe_preserve_order(raw)
                filtered = _filter_available_providers(chain)
                if filtered:
                    if filtered != chain:
                        logger.info(
                            "Filtered injected fallback chain to configured providers. requested=%s usable=%s",
                            chain,
                            filtered,
                        )
                    else:
                        logger.info(
                            "Using fallback chain from injected configuration: %s",
                            filtered,
                        )
                    return filtered

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
        configured: set[str] = set()

        # Candidate providers from injected configuration data (if present).
        candidates: set[str] = set()
        if self._config_cache:
            if 'providers' in self._config_cache:
                candidates.update(str(k).strip().lower() for k in self._config_cache['providers'].keys())
            if 'api_keys' in self._config_cache:
                candidates.update(str(k).strip().lower() for k in self._config_cache['api_keys'].keys())

        # Candidate providers from environment variables.
        for provider in ['openai', 'anthropic', 'openrouter', 'deepseek', 'gemini']:
            env_value = os.getenv(f'{provider.upper()}_API_KEY')
            if not env_value:
                continue
            if self._is_placeholder_value(env_value):
                continue
            candidates.add(provider)

        # Only consider a provider "configured" if it can supply a valid api_key
        # after applying the full resolution logic (config cache -> env -> defaults).
        for provider in candidates:
            try:
                config = self._get_provider_config_dict(provider)
            except Exception:
                continue
            api_key = config.get("api_key")
            if not api_key:
                continue
            if isinstance(api_key, str) and self._is_placeholder_value(api_key):
                continue
            configured.add(provider)

        return list(configured)

    def get_available_providers_by_priority(self) -> List[str]:
        """Get available providers ordered by priority and quality.

        Returns:
            List[str]: List of available provider names in priority order
        """
        # Define priority order based on quality and capabilities
        priority_order = ['openai', 'anthropic', 'openrouter', 'deepseek', 'gemini']
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

        for provider in ['openai', 'anthropic', 'openrouter', 'deepseek', 'gemini']:
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
        logger.debug("Configuration cache refreshed from environment")
