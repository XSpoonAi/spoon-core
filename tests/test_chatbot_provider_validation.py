from unittest.mock import Mock, patch

from spoon_ai.chat import ChatBot


class _FakeConfigManager:
    def __init__(self, provider: str, default_base_url: str, api_key: str):
        self.provider = provider
        self.default_base_url = default_base_url
        self.api_key = api_key

    def _get_provider_config_dict(self, provider_name: str):
        assert provider_name == self.provider
        return {
            "api_key": self.api_key,
            "base_url": self.default_base_url,
            "model": "test-model",
        }

    def _get_provider_defaults(self, provider_name: str):
        assert provider_name == self.provider
        return {"base_url": self.default_base_url}

    def list_configured_providers(self):
        return [self.provider]


class _FakeRuntimeConfigManager:
    def __init__(self):
        self._provider_configs = {}


def _make_llm_manager():
    manager = Mock()
    manager.config_manager = _FakeRuntimeConfigManager()
    return manager


def test_api_key_validation_runs_on_default_base_url():
    # anthropic expects prefix sk-ant-; use an OpenAI-style key to force mismatch
    fake_cfg = _FakeConfigManager(
        provider="anthropic",
        default_base_url="https://api.anthropic.com",
        api_key="sk-test-openai-key",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=_make_llm_manager()):
        with patch("spoon_ai.llm.config.ConfigurationManager", return_value=fake_cfg):
            with patch.object(ChatBot, "_validate_provider_api_key_match", return_value=False) as validate_mock:
                ChatBot(use_llm_manager=True, llm_provider="anthropic")

    # For provider default base URL, validation must run
    assert validate_mock.call_count == 1


def test_api_key_mismatch_skipped_on_custom_base_url():
    # Same mismatched key as above, but custom base_url should bypass prefix validation
    fake_cfg = _FakeConfigManager(
        provider="anthropic",
        default_base_url="https://api.anthropic.com",
        api_key="sk-test-openai-key",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=_make_llm_manager()):
        with patch("spoon_ai.llm.config.ConfigurationManager", return_value=fake_cfg):
            bot = ChatBot(
                use_llm_manager=True,
                llm_provider="anthropic",
                base_url="https://veithly-cliproxyapi.hf.space/v1",
            )

    assert bot.llm_provider == "anthropic"
