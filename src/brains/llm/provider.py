from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...
