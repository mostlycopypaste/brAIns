from __future__ import annotations

import json

from brains.llm.provider import LLMResponse


class FakeLLMProvider:
    """Deterministic LLM provider for testing.

    Returns structured QueryPlan-shaped responses when the system message
    contains 'query planner', and a generic text response otherwise.
    """

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"]
            elif msg["role"] == "user":
                user_content += msg["content"]

        if "query planner" in system_content.lower():
            content = self._plan_response(user_content)
        elif "synthesize" in system_content.lower():
            content = self._synthesis_response(user_content)
        else:
            content = json.dumps(
                {"response": "Fake LLM response", "input_summary": user_content[:100]}
            )

        return LLMResponse(
            content=content,
            model="fake",
            prompt_tokens=len(system_content + user_content) // 4,
            completion_tokens=len(content) // 4,
        )

    def _plan_response(self, user_query: str) -> str:
        query_lower = user_query.lower()

        queries = []
        if any(
            kw in query_lower
            for kw in ["company", "companies", "founded", "employee", "sector"]
        ):
            queries.append(
                {
                    "source": "sql",
                    "query": "SELECT * FROM companies WHERE 1=1",
                    "reasoning": "Question is about company data available in the relational store",
                }
            )
        if any(
            kw in query_lower
            for kw in ["what is", "explain", "concept", "how does", "ai", "machine learning"]
        ):
            queries.append(
                {
                    "source": "vector",
                    "query": user_query,
                    "reasoning": "Question is conceptual and benefits from semantic search",
                }
            )
        if any(
            kw in query_lower
            for kw in ["related", "relationship", "connected", "uses", "built with"]
        ):
            queries.append(
                {
                    "source": "graph",
                    "query": user_query,
                    "reasoning": "Question is about relationships between entities",
                }
            )

        if not queries:
            queries.append(
                {
                    "source": "sql",
                    "query": "SELECT * FROM companies LIMIT 5",
                    "reasoning": "Default routing to relational store",
                }
            )

        return json.dumps({"queries": queries})

    def _synthesis_response(self, context: str) -> str:
        return json.dumps(
            {
                "answer": f"Based on the available data: {context[:200]}",
                "confidence": 0.85,
            }
        )
