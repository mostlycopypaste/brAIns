from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "BRAINS_"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Query defaults
    default_context_budget: int = 4000
    default_response_format: str = "structured"
