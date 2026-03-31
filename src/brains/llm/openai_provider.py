from __future__ import annotations

from openai import OpenAI

from brains.llm.provider import LLMResponse


class OpenAIProvider:
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
