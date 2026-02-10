from unittest.mock import Mock, patch

import pytest

from spoon_ai.chat import ChatBot
from spoon_ai.llm.errors import ConfigurationError


class _FakeConfigMissingKey:
    def _get_provider_config_dict(self, provider_name: str):
        assert provider_name == "gemini"
        return {
            "api_key": "",  # explicit missing key
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model": "gemini-3-flash-preview",
        }

    def _get_provider_defaults(self, provider_name: str):
        assert provider_name == "gemini"
        return {"base_url": "https://generativelanguage.googleapis.com/v1beta"}

    def list_configured_providers(self):
        return ["gemini"]


class _FakeRuntimeConfigManager:
    def __init__(self):
        self._provider_configs = {}


def _make_llm_manager():
    manager = Mock()
    manager.config_manager = _FakeRuntimeConfigManager()
    return manager


def test_explicit_provider_missing_key_raises_instead_of_fallback():
    with patch("spoon_ai.chat.get_llm_manager", return_value=_make_llm_manager()):
        with patch("spoon_ai.llm.config.ConfigurationManager", return_value=_FakeConfigMissingKey()):
            with pytest.raises(ConfigurationError) as exc:
                ChatBot(use_llm_manager=True, llm_provider="gemini")

    assert "Missing API key for explicitly requested provider 'gemini'" in str(exc.value)
