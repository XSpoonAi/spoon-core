from __future__ import annotations

from spoon_ai.llm.config import ConfigurationManager


def test_cypher_proxy_token_configures_openrouter_without_provider_key(monkeypatch):
    monkeypatch.setenv("CYPHER_LLM_PROXY_ENABLED", "true")
    monkeypatch.setenv("CYPHER_LLM_PROXY_TOKEN", "cypher_sbx_v1_test")
    monkeypatch.setenv("SPOON_BOT_DEFAULT_PROVIDER", "openrouter")
    monkeypatch.setenv("SPOON_BOT_DEFAULT_BASE_URL", "https://api.cypher.local/api/v1/llm-proxy/openrouter/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-real-provider-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    manager = ConfigurationManager()

    config = manager._get_provider_config_dict("openrouter")
    assert config["api_key"] == "cypher_sbx_v1_test"
    assert config["base_url"] == "https://api.cypher.local/api/v1/llm-proxy/openrouter/api/v1"
    assert manager.get_default_provider() == "openrouter"
    assert "openrouter" in manager.get_available_providers_by_priority()
    assert "openrouter" in manager.list_configured_providers()