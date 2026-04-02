from __future__ import annotations

import json
import logging
from typing import Any

from brains.llm.provider import LLMProvider
from brains.models import QueryMetadata, QueryRequest, QueryResponse, SourceResult
from brains.sources.base import DataSource

logger = logging.getLogger(__name__)


INTERPRETATION_SYSTEM_PROMPT = (
    "You are a query planner for a data service. "
    "Given a user query, decide which data sources to consult "
    "and what queries to run.\n\n"
    "Available data sources:\n"
    "{source_descriptions}\n\n"
    "Return a JSON object with a \"queries\" array. "
    "Each query object must have:\n"
    "- \"source\": the data source name (one of: {source_names})\n"
    "- \"query\": the specific query to run "
    "(SQL for sql source, natural text for vector source, "
    "JSON operation for graph source)\n"
    "- \"reasoning\": why this source and query are appropriate\n\n"
    "Route queries to the most relevant sources. "
    "Use multiple sources when the question benefits "
    "from cross-referencing."
)


SYNTHESIS_SYSTEM_PROMPT_STRUCTURED = (
    "You are a data synthesis assistant. "
    "Synthesize the following query results into a structured response. "
    "Include source attribution.\n\n"
    "Return a JSON object with:\n"
    "- \"answer\": a JSON-formatted string organizing the synthesized data "
    "with clear keys and structure (the value must be a string, not a nested object)\n"
    "- \"confidence\": overall confidence score between 0 and 1"
)

SYNTHESIS_SYSTEM_PROMPT_NARRATIVE = (
    "You are a data synthesis assistant. "
    "Synthesize the following query results into a clear, "
    "natural language narrative. Include source attribution.\n\n"
    "Return a JSON object with:\n"
    "- \"answer\": a fluent prose summary of the results, "
    "written for a human reader\n"
    "- \"confidence\": overall confidence score between 0 and 1"
)


class QueryEngine:
    def __init__(
        self,
        llm: LLMProvider,
        sources: dict[str, DataSource],
        model: str,
        default_context_budget: int = 4000,
    ) -> None:
        self._llm = llm
        self._sources = sources
        self._model = model
        self._default_context_budget = default_context_budget

    @property
    def sources(self) -> dict[str, DataSource]:
        return self._sources

    def execute(self, request: QueryRequest) -> QueryResponse:
        context_budget = request.context_budget or self._default_context_budget

        query_plan, interpretation_model = self._interpret(request)

        all_results: list[SourceResult] = []
        sources_consulted: list[str] = []
        total_tokens = 0

        for planned_query in query_plan:
            source_name = planned_query["source"]

            if request.sources and source_name not in request.sources:
                continue

            if source_name not in self._sources:
                logger.warning("Query plan referenced unknown source: %s", source_name)
                continue

            source = self._sources[source_name]
            query_params = self._build_query_params(source_name, planned_query["query"])
            raw_results = source.query(query_params)

            if raw_results:
                sources_consulted.append(source_name)
                source_result = SourceResult(
                    source=source_name,
                    confidence=max(r.score for r in raw_results),
                    data=[r.data for r in raw_results],
                    query_used=str(planned_query["query"]),
                )
                all_results.append(source_result)
                total_tokens += self._estimate_tokens(source_result)

        answer = None
        synthesis_model = None

        if request.response_format != "raw" and all_results:
            synthesis, synthesis_model = self._synthesize(
                request.query, all_results, context_budget, request.response_format
            )
            answer = synthesis.get("answer")

        return QueryResponse(
            answer=answer,
            sources_consulted=sources_consulted,
            results=all_results,
            metadata=QueryMetadata(
                total_results=sum(len(r.data) for r in all_results),
                context_tokens_used=total_tokens,
                interpretation_model=interpretation_model,
                synthesis_model=synthesis_model,
            ),
        )

    def _interpret(self, request: QueryRequest) -> tuple[list[dict[str, Any]], str]:
        source_descriptions = "\n".join(
            f"- {name}: {source.describe().description}"
            for name, source in self._sources.items()
        )
        source_names = ", ".join(self._sources.keys())

        system_prompt = INTERPRETATION_SYSTEM_PROMPT.format(
            source_descriptions=source_descriptions,
            source_names=source_names,
        )

        response = self._llm.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query},
            ],
            model=self._model,
            temperature=0.0,
            response_format="json",
        )

        # Strip markdown code fences if present (some LLMs wrap JSON in ```json ... ```)
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        try:
            plan = json.loads(content)
            return plan.get("queries", []), response.model
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(
                "Failed to parse query plan from LLM response. Error: %s, Response: %s",
                e,
                response.content[:500],
            )
            fallback = [
                {"source": "sql", "query": request.query, "reasoning": "fallback"}
            ]
            return fallback, response.model

    def _build_query_params(self, source_name: str, query: str | dict) -> dict[str, Any]:
        if source_name == "sql":
            return {"sql": query if isinstance(query, str) else str(query)}
        elif source_name == "vector":
            return {"text": query if isinstance(query, str) else str(query), "top_k": 5}
        elif source_name == "graph":
            if isinstance(query, dict):
                return query
            try:
                return json.loads(query)
            except (json.JSONDecodeError, TypeError):
                return {"operation": "neighbors", "node": str(query)}
        return {"query": query}

    def _synthesize(
        self,
        original_query: str,
        results: list[SourceResult],
        context_budget: int,
        response_format: str = "structured",
    ) -> tuple[dict[str, Any], str]:
        results_text = json.dumps(
            [{"source": r.source, "data": r.data[:10]} for r in results],
            indent=2,
            default=str,
        )

        max_chars = context_budget * 4
        if len(results_text) > max_chars:
            results_text = results_text[:max_chars] + "\n... (truncated)"

        system_prompt = (
            SYNTHESIS_SYSTEM_PROMPT_NARRATIVE
            if response_format == "narrative"
            else SYNTHESIS_SYSTEM_PROMPT_STRUCTURED
        )

        response = self._llm.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Original query: {original_query}\n\nResults:\n{results_text}",
                },
            ],
            model=self._model,
            temperature=0.0,
            response_format="json",
        )

        # Strip markdown code fences if present
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        try:
            return json.loads(content), response.model
        except (json.JSONDecodeError, AttributeError):
            return {"answer": response.content, "confidence": 0.5}, response.model

    def _estimate_tokens(self, result: SourceResult) -> int:
        text = json.dumps(result.data, default=str)
        return len(text) // 4
