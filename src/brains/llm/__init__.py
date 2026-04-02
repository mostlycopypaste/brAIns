from __future__ import annotations

from brains.llm.fake_provider import FakeLLMProvider
from brains.llm.ollama_provider import OllamaProvider
from brains.llm.openai_provider import OpenAIProvider
from brains.llm.provider import LLMProvider, LLMResponse


def create_provider(
    provider_name: str, api_key: str = "", ollama_base_url: str = ""
) -> LLMProvider:
    if provider_name == "fake":
        return FakeLLMProvider()
    if provider_name == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required when using the openai provider")
        return OpenAIProvider(api_key=api_key)
    if provider_name == "ollama":
        return OllamaProvider(base_url=ollama_base_url or "http://localhost:11434/v1")
    raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = ["LLMProvider", "LLMResponse", "create_provider"]
