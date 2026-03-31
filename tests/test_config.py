import pytest


def test_default_config_loads():
    """Config loads with sensible defaults when no env vars are set."""
    from brains.config import Settings

    settings = Settings(openai_api_key="test-key")
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_provider == "openai"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.default_context_budget == 4000


def test_config_from_env(monkeypatch):
    """Config reads from environment variables with BRAINS_ prefix."""
    monkeypatch.setenv("BRAINS_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("BRAINS_LLM_PROVIDER", "fake")
    monkeypatch.setenv("BRAINS_DEFAULT_CONTEXT_BUDGET", "8000")
    monkeypatch.setenv("BRAINS_OPENAI_API_KEY", "sk-test-123")
    from brains.config import Settings

    settings = Settings()
    assert settings.llm_model == "gpt-4o"
    assert settings.llm_provider == "fake"
    assert settings.default_context_budget == 8000
    assert settings.openai_api_key == "sk-test-123"
