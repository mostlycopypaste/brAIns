# brAIns MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use trycycle-executing to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MVP service where AI agents can query curated, specialized data sources via REST API. An LLM interprets incoming requests, routes them to appropriate data backends (semantic search, SQL, vector search), and optionally synthesizes results before returning them. The service prioritizes accuracy and context management over speed. Deployed via Docker and docker-compose with sample/stub data sources demonstrating the pattern.

**Architecture:** A Python FastAPI service with three layers: (1) a REST API that accepts natural-language or structured queries from AI agents, (2) an LLM orchestration layer that interprets requests, selects data sources, formulates backend-specific queries, and synthesizes results, and (3) a pluggable data source layer with a uniform interface. The MVP ships with three stub data sources — a SQLite relational store, an in-memory vector store using numpy for cosine similarity, and an in-memory graph store using networkx — each pre-loaded with sample domain data. A thin LLM provider abstraction defaults to OpenAI but is swappable via configuration. Docker and docker-compose wrap the service for deployment.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, Pydantic v2, OpenAI SDK, numpy, networkx, SQLite (stdlib), Docker, docker-compose, pytest, httpx (test client)

---

## Design decisions

### Python with FastAPI over Go or Node

The user has no language preference. Python is the strongest choice for this MVP because: (1) the richest LLM SDK ecosystem lives in Python, (2) FastAPI's automatic OpenAPI docs give AI agent consumers a self-describing API for free, (3) numpy and networkx are Python-native and avoid FFI overhead for the vector and graph stubs, (4) the user cares about accuracy and context management, not raw throughput — Python's performance is more than sufficient. Go would be a reasonable choice for a production rewrite if throughput becomes a bottleneck, but for an MVP proving the pattern, Python is faster to build and easier to iterate on.

### Thin LLM abstraction over litellm

litellm is a popular multi-provider abstraction, but it's a heavy dependency with its own opinions about retries, caching, and error handling. For an MVP that defaults to OpenAI and needs provider-agnostic design without overcomplication, a thin protocol-based abstraction is cleaner: define a `LLMProvider` protocol with a single `complete(messages, model, temperature, response_format)` method, ship an `OpenAIProvider` implementation, and swap providers by changing one config value. This keeps the dependency surface small and the abstraction honest. Adding a `ClaudeProvider` or `OllamaProvider` later is a single-file addition.

### Request interpretation via structured LLM output, not free-form

When an agent sends a query, the LLM must decide which data sources to consult and what queries to run. Using structured output (Pydantic models as response_format) makes this deterministic and parseable: the LLM returns a `QueryPlan` with typed fields for each data source query. This avoids regex parsing of free-form LLM output and catches malformed plans at the Pydantic validation layer. The system prompt for interpretation will include the schema of available data sources and their query capabilities so the LLM can make informed routing decisions.

### Three stub data sources demonstrating the pattern

The MVP needs enough variety to prove the multi-source routing pattern without building real integrations:

1. **SQLite relational store** — a `companies` table with tech company data (name, sector, founded year, headquarters, employee count, description). Demonstrates SQL query routing. Uses Python's built-in sqlite3 module — zero dependencies.

2. **In-memory vector store** — pre-embedded text chunks about AI/ML concepts using OpenAI embeddings at startup. Demonstrates semantic search with cosine similarity via numpy. Chunks are loaded from a JSON fixture file so the embedding step is reproducible.

3. **In-memory graph store** — a knowledge graph of technology relationships (languages, frameworks, companies, concepts) using networkx. Demonstrates graph traversal queries (neighbors, paths, subgraphs).

Each data source implements a `DataSource` protocol with `query(params) -> list[DataResult]` and `describe() -> DataSourceSchema`. The `describe()` method returns structured metadata the LLM uses for routing decisions.

### Context management strategy

The user specifically called out context management on the service side. The MVP implements three mechanisms:

1. **Result ranking and truncation** — when multiple sources return results, they're scored by relevance (the LLM assigns relevance scores during synthesis) and truncated to fit within a configurable context budget. This prevents overwhelming the requesting agent with too much data.

2. **Source attribution** — every result includes metadata about which data source produced it and with what confidence, so the requesting agent can make informed decisions about data quality.

3. **Response shaping** — the LLM synthesis step can be configured to return raw results, a summarized narrative, or structured data, depending on what the requesting agent asks for. This is controlled by an optional `response_format` field in the request.

### Query flow

```
Agent request (REST)
  -> Parse & validate (Pydantic)
  -> LLM interprets request, produces QueryPlan
  -> Execute queries against selected data sources (parallel)
  -> Collect results with source attribution
  -> LLM synthesizes results (optional, based on request)
  -> Return structured response with metadata
```

The two-LLM-call pattern (interpret + synthesize) is intentional: interpretation must happen before data source queries, and synthesis must happen after. The synthesis step is optional — agents can request raw results if they prefer to do their own reasoning.

### Docker setup

A single Dockerfile for the FastAPI service. docker-compose.yml exposes the service on port 8000. The SQLite database and vector embeddings are generated at container startup from fixture files, so the container is self-contained with no external dependencies except an OpenAI API key (passed via environment variable). A health check endpoint at `/health` enables docker-compose health monitoring.

### API design

Three endpoints:

1. `POST /query` — the main endpoint. Accepts a natural-language or structured query, runs the full interpret-query-synthesize pipeline, returns results with metadata.

2. `GET /sources` — lists available data sources and their schemas. Useful for AI agents to understand what data is available before querying.

3. `GET /health` — health check for docker-compose and load balancers.

The `/query` endpoint accepts:
```json
{
  "query": "What companies in the AI sector were founded after 2010?",
  "sources": ["sql", "vector", "graph"],  // optional filter
  "response_format": "structured",  // "raw" | "structured" | "narrative"
  "context_budget": 4000  // optional max tokens in response
}
```

And returns:
```json
{
  "answer": "...",
  "sources_consulted": ["sql"],
  "results": [
    {
      "source": "sql",
      "confidence": 0.95,
      "data": [...],
      "query_used": "SELECT * FROM companies WHERE sector = 'AI' AND founded > 2010"
    }
  ],
  "metadata": {
    "total_results": 3,
    "context_tokens_used": 1200,
    "interpretation_model": "gpt-4o-mini",
    "synthesis_model": "gpt-4o-mini"
  }
}
```

### LLM model selection

Default to `gpt-4o-mini` for both interpretation and synthesis. It's cheap, fast, and accurate enough for structured output. The model is configurable via environment variable (`BRAINS_LLM_MODEL`). For the MVP, using the same model for both steps is simpler and sufficient.

### Testing with fake LLM provider

The testing strategy requires integration tests against the running service with a fake LLM provider. The `LLMProvider` protocol makes this clean: a `FakeLLMProvider` returns canned `QueryPlan` responses for known inputs and a default response for unknown inputs. This lets integration tests exercise the full pipeline without hitting the OpenAI API. Container smoke tests use the real service with the fake provider injected via environment variable (`BRAINS_LLM_PROVIDER=fake`).

### Project structure

```
brAIns/
  src/
    brains/
      __init__.py
      main.py              # FastAPI app, startup, endpoints
      config.py            # Settings via pydantic-settings
      models.py            # Request/response Pydantic models
      query_engine.py      # Orchestrates interpret -> query -> synthesize
      llm/
        __init__.py
        provider.py        # LLMProvider protocol
        openai_provider.py # OpenAI implementation
        fake_provider.py   # Fake for testing
      sources/
        __init__.py
        base.py            # DataSource protocol, DataResult
        sql_source.py      # SQLite data source
        vector_source.py   # Numpy vector store
        graph_source.py    # NetworkX graph store
      data/
        companies.json     # SQL fixture data
        knowledge.json     # Vector store text chunks
        graph.json         # Graph edges and nodes
  tests/
    __init__.py
    conftest.py            # Shared fixtures, test client
    test_api.py            # HTTP endpoint tests
    test_query_engine.py   # Query pipeline unit tests
    test_sources.py        # Data source unit tests
    test_llm.py            # LLM provider tests
    test_smoke.py          # Container smoke tests
  Dockerfile
  docker-compose.yml
  pyproject.toml
  README.md
```

Feature-oriented organization within `src/brains/` with `llm/` and `sources/` as the two major subsystems. `data/` holds fixture files, not code. Tests are in a top-level `tests/` directory adjacent to `src/` following pytest conventions.

---

## File structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/brains/__init__.py` | Package marker |
| Create | `src/brains/main.py` | FastAPI app with endpoints, startup/shutdown lifecycle |
| Create | `src/brains/config.py` | Pydantic Settings for environment-driven configuration |
| Create | `src/brains/models.py` | Request/response Pydantic models (QueryRequest, QueryResponse, etc.) |
| Create | `src/brains/query_engine.py` | Core orchestrator: interpret, query, synthesize |
| Create | `src/brains/llm/__init__.py` | Package marker |
| Create | `src/brains/llm/provider.py` | LLMProvider protocol definition |
| Create | `src/brains/llm/openai_provider.py` | OpenAI SDK implementation of LLMProvider |
| Create | `src/brains/llm/fake_provider.py` | Fake LLM provider for testing |
| Create | `src/brains/sources/__init__.py` | Package marker |
| Create | `src/brains/sources/base.py` | DataSource protocol, DataResult model, DataSourceSchema |
| Create | `src/brains/sources/sql_source.py` | SQLite-backed relational data source |
| Create | `src/brains/sources/vector_source.py` | Numpy-based vector similarity search |
| Create | `src/brains/sources/graph_source.py` | NetworkX-based graph traversal |
| Create | `src/brains/data/companies.json` | Fixture: tech company records |
| Create | `src/brains/data/knowledge.json` | Fixture: AI/ML knowledge chunks with pre-computed embeddings |
| Create | `src/brains/data/graph.json` | Fixture: technology relationship graph |
| Create | `tests/__init__.py` | Package marker |
| Create | `tests/conftest.py` | Shared pytest fixtures, test client factory |
| Create | `tests/test_api.py` | HTTP integration tests for all endpoints |
| Create | `tests/test_query_engine.py` | Unit tests for the query pipeline |
| Create | `tests/test_sources.py` | Unit tests for each data source |
| Create | `tests/test_llm.py` | Unit tests for LLM provider abstraction |
| Create | `tests/test_smoke.py` | Container smoke tests (docker-compose up, hit endpoints) |
| Create | `Dockerfile` | Multi-stage Docker build |
| Create | `docker-compose.yml` | Service orchestration with health checks |
| Create | `pyproject.toml` | Project metadata, dependencies, tool config |

---

### Task 1: Project scaffolding, configuration, and models

**Files:**
- Create: `pyproject.toml`
- Create: `src/brains/__init__.py`
- Create: `src/brains/config.py`
- Create: `src/brains/models.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write failing test for configuration loading**

```python
# tests/test_config.py
import os
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_config.py -v`
Expected: FAIL — module `brains` not found

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "brains"
version = "0.1.0"
description = "AI-augmented curated data service"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "openai>=1.50.0",
    "numpy>=2.0",
    "networkx>=3.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
    "ruff>=0.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/brains"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

- [ ] **Step 4: Create config.py**

```python
# src/brains/config.py
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
```

- [ ] **Step 5: Create models.py**

```python
# src/brains/models.py
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language or structured query")
    sources: list[str] | None = Field(
        default=None,
        description="Optional filter for which data sources to consult",
    )
    response_format: str = Field(
        default="structured",
        description="How to format the response: raw, structured, or narrative",
    )
    context_budget: int | None = Field(
        default=None,
        description="Maximum tokens in the response (overrides server default)",
    )


class SourceResult(BaseModel):
    source: str
    confidence: float
    data: list[dict]
    query_used: str


class QueryMetadata(BaseModel):
    total_results: int
    context_tokens_used: int
    interpretation_model: str
    synthesis_model: str | None = None


class QueryResponse(BaseModel):
    answer: str | None = None
    sources_consulted: list[str]
    results: list[SourceResult]
    metadata: QueryMetadata


class DataSourceInfo(BaseModel):
    name: str
    description: str
    capabilities: list[str]
    sample_queries: list[str]


class SourcesResponse(BaseModel):
    sources: list[DataSourceInfo]


class HealthResponse(BaseModel):
    status: str
    sources_loaded: int
    llm_provider: str
```

- [ ] **Step 6: Create package markers**

Create empty `src/brains/__init__.py` and `tests/__init__.py`.

- [ ] **Step 7: Install the project in dev mode and run the test**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && pip install -e ".[dev]" && python3 -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 8: Refactor and verify**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add pyproject.toml src/ tests/
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: project scaffolding with config, models, and pyproject.toml"
```

---

### Task 2: LLM provider abstraction and implementations

**Files:**
- Create: `src/brains/llm/__init__.py`
- Create: `src/brains/llm/provider.py`
- Create: `src/brains/llm/openai_provider.py`
- Create: `src/brains/llm/fake_provider.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing test for LLM provider protocol**

```python
# tests/test_llm.py
import json
import pytest
from brains.llm.provider import LLMProvider, LLMResponse
from brains.llm.fake_provider import FakeLLMProvider


class TestFakeLLMProvider:
    def test_fake_provider_implements_protocol(self):
        provider = FakeLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_fake_provider_returns_canned_response(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4o-mini",
        )
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.model == "fake"

    def test_fake_provider_returns_structured_query_plan(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[
                {"role": "system", "content": "You are a query planner."},
                {"role": "user", "content": "What AI companies were founded after 2015?"},
            ],
            model="gpt-4o-mini",
            response_format="json",
        )
        parsed = json.loads(response.content)
        assert "queries" in parsed
        assert isinstance(parsed["queries"], list)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_llm.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Create the LLM provider protocol**

```python
# src/brains/llm/provider.py
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
```

- [ ] **Step 4: Create the fake provider**

```python
# src/brains/llm/fake_provider.py
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
            content = json.dumps({"response": "Fake LLM response", "input_summary": user_content[:100]})

        return LLMResponse(
            content=content,
            model="fake",
            prompt_tokens=len(system_content + user_content) // 4,
            completion_tokens=len(content) // 4,
        )

    def _plan_response(self, user_query: str) -> str:
        query_lower = user_query.lower()

        queries = []
        # Route to SQL for structured/tabular questions
        if any(kw in query_lower for kw in ["company", "companies", "founded", "employee", "sector"]):
            queries.append({
                "source": "sql",
                "query": "SELECT * FROM companies WHERE 1=1",
                "reasoning": "Question is about company data available in the relational store",
            })
        # Route to vector for conceptual/knowledge questions
        if any(kw in query_lower for kw in ["what is", "explain", "concept", "how does", "ai", "machine learning"]):
            queries.append({
                "source": "vector",
                "query": user_query,
                "reasoning": "Question is conceptual and benefits from semantic search",
            })
        # Route to graph for relationship questions
        if any(kw in query_lower for kw in ["related", "relationship", "connected", "uses", "built with"]):
            queries.append({
                "source": "graph",
                "query": user_query,
                "reasoning": "Question is about relationships between entities",
            })

        # Default to SQL if no keywords matched
        if not queries:
            queries.append({
                "source": "sql",
                "query": "SELECT * FROM companies LIMIT 5",
                "reasoning": "Default routing to relational store",
            })

        return json.dumps({"queries": queries})

    def _synthesis_response(self, context: str) -> str:
        return json.dumps({
            "answer": f"Based on the available data: {context[:200]}",
            "confidence": 0.85,
        })
```

- [ ] **Step 5: Create the OpenAI provider**

```python
# src/brains/llm/openai_provider.py
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
```

- [ ] **Step 6: Create `__init__.py` with factory function**

```python
# src/brains/llm/__init__.py
from __future__ import annotations

from brains.llm.provider import LLMProvider, LLMResponse
from brains.llm.fake_provider import FakeLLMProvider
from brains.llm.openai_provider import OpenAIProvider


def create_provider(provider_name: str, api_key: str = "") -> LLMProvider:
    if provider_name == "fake":
        return FakeLLMProvider()
    if provider_name == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required when using the openai provider")
        return OpenAIProvider(api_key=api_key)
    raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = ["LLMProvider", "LLMResponse", "create_provider"]
```

- [ ] **Step 7: Run the tests**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 8: Refactor and verify full suite**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add src/brains/llm/ tests/test_llm.py
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: LLM provider abstraction with OpenAI and fake implementations"
```

---

### Task 3: Data source protocol and SQLite source

**Files:**
- Create: `src/brains/sources/__init__.py`
- Create: `src/brains/sources/base.py`
- Create: `src/brains/sources/sql_source.py`
- Create: `src/brains/data/companies.json`
- Create: `tests/test_sources.py`

- [ ] **Step 1: Write failing test for the SQL data source**

```python
# tests/test_sources.py
import pytest
from brains.sources.base import DataSource
from brains.sources.sql_source import SQLiteSource


class TestSQLiteSource:
    @pytest.fixture
    def source(self, tmp_path):
        return SQLiteSource(db_path=tmp_path / "test.db")

    def test_implements_protocol(self, source):
        assert isinstance(source, DataSource)

    def test_describe_returns_schema(self, source):
        schema = source.describe()
        assert schema.name == "sql"
        assert "companies" in schema.description.lower()
        assert len(schema.capabilities) > 0

    def test_query_returns_results(self, source):
        results = source.query({"sql": "SELECT * FROM companies LIMIT 3"})
        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert "name" in r.data
            assert r.source == "sql"

    def test_query_with_filter(self, source):
        results = source.query({"sql": "SELECT * FROM companies WHERE sector = 'AI'"})
        assert len(results) > 0
        for r in results:
            assert r.data["sector"] == "AI"

    def test_query_invalid_sql_returns_error(self, source):
        results = source.query({"sql": "DROP TABLE companies"})
        assert len(results) == 0  # write operations are rejected
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_sources.py::TestSQLiteSource -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Create the data source protocol**

```python
# src/brains/sources/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class DataSourceSchema:
    name: str
    description: str
    capabilities: list[str]
    sample_queries: list[str]


@dataclass
class DataResult:
    source: str
    data: dict[str, Any]
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataSource(Protocol):
    def query(self, params: dict[str, Any]) -> list[DataResult]: ...
    def describe(self) -> DataSourceSchema: ...
```

- [ ] **Step 4: Create the companies fixture data**

```json
// src/brains/data/companies.json
[
    {"name": "OpenAI", "sector": "AI", "founded": 2015, "headquarters": "San Francisco, CA", "employees": 3000, "description": "AI research company known for GPT models and ChatGPT"},
    {"name": "Anthropic", "sector": "AI", "founded": 2021, "headquarters": "San Francisco, CA", "employees": 1500, "description": "AI safety company, creator of Claude"},
    {"name": "Google DeepMind", "sector": "AI", "founded": 2010, "headquarters": "London, UK", "employees": 2500, "description": "AI research lab known for AlphaGo and Gemini"},
    {"name": "Meta AI", "sector": "AI", "founded": 2013, "headquarters": "Menlo Park, CA", "employees": 1000, "description": "AI research division of Meta, known for LLaMA models"},
    {"name": "Hugging Face", "sector": "AI", "founded": 2016, "headquarters": "New York, NY", "employees": 400, "description": "Open-source AI platform and model hub"},
    {"name": "Databricks", "sector": "Data", "founded": 2013, "headquarters": "San Francisco, CA", "employees": 7000, "description": "Unified data analytics platform, creators of Apache Spark"},
    {"name": "Snowflake", "sector": "Data", "founded": 2012, "headquarters": "Bozeman, MT", "employees": 6800, "description": "Cloud data warehouse platform"},
    {"name": "Scale AI", "sector": "AI", "founded": 2016, "headquarters": "San Francisco, CA", "employees": 600, "description": "Data labeling and AI infrastructure company"},
    {"name": "Cohere", "sector": "AI", "founded": 2019, "headquarters": "Toronto, Canada", "employees": 450, "description": "Enterprise NLP and large language model provider"},
    {"name": "Mistral AI", "sector": "AI", "founded": 2023, "headquarters": "Paris, France", "employees": 100, "description": "Open-weight large language model company"},
    {"name": "Elastic", "sector": "Search", "founded": 2012, "headquarters": "Mountain View, CA", "employees": 3200, "description": "Search and observability platform based on Elasticsearch"},
    {"name": "Neo4j", "sector": "Database", "founded": 2007, "headquarters": "San Mateo, CA", "employees": 900, "description": "Graph database platform"},
    {"name": "Pinecone", "sector": "AI", "founded": 2019, "headquarters": "New York, NY", "employees": 200, "description": "Vector database for AI applications"},
    {"name": "Weaviate", "sector": "AI", "founded": 2019, "headquarters": "Amsterdam, Netherlands", "employees": 150, "description": "Open-source vector database"},
    {"name": "Stability AI", "sector": "AI", "founded": 2019, "headquarters": "London, UK", "employees": 200, "description": "Open-source generative AI company, known for Stable Diffusion"}
]
```

- [ ] **Step 5: Create the SQLite data source**

```python
# src/brains/sources/sql_source.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from brains.sources.base import DataResult, DataSourceSchema

DATA_DIR = Path(__file__).parent.parent / "data"
ALLOWED_STATEMENTS = {"SELECT"}


class SQLiteSource:
    def __init__(self, db_path: Path | None = None):
        self._db_path = str(db_path) if db_path else ":memory:"
        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row
        self._load_fixture_data()

    def _load_fixture_data(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                name TEXT PRIMARY KEY,
                sector TEXT NOT NULL,
                founded INTEGER NOT NULL,
                headquarters TEXT NOT NULL,
                employees INTEGER NOT NULL,
                description TEXT NOT NULL
            )
        """)
        fixture_path = DATA_DIR / "companies.json"
        if fixture_path.exists():
            with open(fixture_path) as f:
                companies = json.load(f)
            for company in companies:
                cursor.execute(
                    "INSERT OR IGNORE INTO companies VALUES (?, ?, ?, ?, ?, ?)",
                    (company["name"], company["sector"], company["founded"],
                     company["headquarters"], company["employees"], company["description"]),
                )
        self._connection.commit()

    def query(self, params: dict[str, Any]) -> list[DataResult]:
        sql = params.get("sql", "")
        statement_type = sql.strip().split()[0].upper() if sql.strip() else ""
        if statement_type not in ALLOWED_STATEMENTS:
            return []

        try:
            cursor = self._connection.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
        except sqlite3.Error:
            return []

        return [
            DataResult(
                source="sql",
                data=dict(row),
                score=1.0,
                metadata={"query": sql},
            )
            for row in rows
        ]

    def describe(self) -> DataSourceSchema:
        return DataSourceSchema(
            name="sql",
            description="SQLite relational database with tech companies data. Table: companies (name, sector, founded, headquarters, employees, description).",
            capabilities=["SQL SELECT queries", "Filtering", "Aggregation", "Sorting"],
            sample_queries=[
                "SELECT * FROM companies WHERE sector = 'AI'",
                "SELECT name, founded FROM companies WHERE founded > 2015 ORDER BY founded",
                "SELECT sector, COUNT(*) as count FROM companies GROUP BY sector",
            ],
        )
```

- [ ] **Step 6: Create `sources/__init__.py`**

```python
# src/brains/sources/__init__.py
from brains.sources.base import DataSource, DataResult, DataSourceSchema
from brains.sources.sql_source import SQLiteSource

__all__ = ["DataSource", "DataResult", "DataSourceSchema", "SQLiteSource"]
```

- [ ] **Step 7: Run the tests**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_sources.py::TestSQLiteSource -v`
Expected: PASS

- [ ] **Step 8: Refactor and verify full suite**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add src/brains/sources/ src/brains/data/companies.json tests/test_sources.py
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: data source protocol and SQLite source with company fixtures"
```

---

### Task 4: Vector similarity data source

**Files:**
- Create: `src/brains/sources/vector_source.py`
- Create: `src/brains/data/knowledge.json`
- Modify: `src/brains/sources/__init__.py`
- Modify: `tests/test_sources.py`

- [ ] **Step 1: Write failing test for the vector source**

```python
# In tests/test_sources.py, add:
from brains.sources.vector_source import VectorSource


class TestVectorSource:
    @pytest.fixture
    def source(self):
        return VectorSource()

    def test_implements_protocol(self, source):
        assert isinstance(source, DataSource)

    def test_describe_returns_schema(self, source):
        schema = source.describe()
        assert schema.name == "vector"
        assert len(schema.capabilities) > 0

    def test_query_returns_ranked_results(self, source):
        results = source.query({"text": "neural networks and deep learning", "top_k": 3})
        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert r.source == "vector"
            assert 0.0 <= r.score <= 1.0
            assert "text" in r.data
        # Results should be sorted by descending score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_empty_text_returns_empty(self, source):
        results = source.query({"text": "", "top_k": 3})
        assert results == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_sources.py::TestVectorSource -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create the knowledge fixture data with pre-computed embeddings**

Create `src/brains/data/knowledge.json` — a list of objects, each with `id`, `text`, `category`, and `embedding` (a pre-computed 64-dimensional float vector for the MVP, avoiding the need to call the OpenAI embeddings API at startup). Use 64 dimensions instead of 1536 to keep the fixture file small. The vector source normalizes all vectors at load time, so the fixture embeddings just need to be directionally meaningful for testing.

Generate the fixture by hand: assign each chunk a random-but-thematically-clustered embedding vector where chunks in the same category have similar vectors. Include 15-20 chunks covering topics like: transformer architecture, attention mechanisms, gradient descent, reinforcement learning, neural networks, embeddings, tokenization, fine-tuning, prompt engineering, RAG, vector databases, knowledge graphs, etc.

```json
[
    {
        "id": "chunk_01",
        "text": "Transformer architecture uses self-attention mechanisms to process sequences in parallel rather than sequentially. This enables much faster training on modern hardware compared to recurrent neural networks.",
        "category": "architecture",
        "embedding": [0.8, 0.7, 0.1, ...]
    }
]
```

The actual embedding values will be deterministic pseudo-random vectors seeded by category, so that semantic similarity within categories is preserved during testing. The implementation should generate these at module load if the fixture doesn't include them.

- [ ] **Step 4: Create the vector source**

```python
# src/brains/sources/vector_source.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from brains.sources.base import DataResult, DataSourceSchema

DATA_DIR = Path(__file__).parent.parent / "data"


class VectorSource:
    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._load_fixture_data()

    def _load_fixture_data(self) -> None:
        fixture_path = DATA_DIR / "knowledge.json"
        if not fixture_path.exists():
            return

        with open(fixture_path) as f:
            self._chunks = json.load(f)

        if not self._chunks:
            return

        raw_embeddings = [chunk["embedding"] for chunk in self._chunks]
        matrix = np.array(raw_embeddings, dtype=np.float32)
        # Normalize to unit vectors for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = matrix / norms

    def _embed_query(self, text: str) -> np.ndarray:
        """Create a simple bag-of-words style embedding for the query.

        For the MVP, this uses a deterministic hash-based approach rather than
        calling an external embedding API. Each word in the query hashes to
        dimensions in the embedding vector, creating a sparse but reproducible
        representation. This is sufficient for demonstrating the retrieval
        pattern — a production system would use a real embedding model.
        """
        dim = self._embeddings.shape[1] if self._embeddings is not None else 64
        vec = np.zeros(dim, dtype=np.float32)
        for word in text.lower().split():
            index = hash(word) % dim
            vec[index] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def query(self, params: dict[str, Any]) -> list[DataResult]:
        text = params.get("text", "")
        top_k = params.get("top_k", 5)

        if not text or self._embeddings is None or len(self._chunks) == 0:
            return []

        query_vec = self._embed_query(text)
        similarities = self._embeddings @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(DataResult(
                source="vector",
                data={"text": chunk["text"], "category": chunk.get("category", "unknown"), "id": chunk["id"]},
                score=score,
                metadata={"similarity": score},
            ))
        return results

    def describe(self) -> DataSourceSchema:
        return DataSourceSchema(
            name="vector",
            description="In-memory vector store with AI/ML knowledge chunks. Supports semantic similarity search over pre-embedded text passages.",
            capabilities=["Semantic similarity search", "Top-K retrieval", "Category filtering"],
            sample_queries=[
                "How do transformer models work?",
                "What is reinforcement learning?",
                "Explain the attention mechanism",
            ],
        )
```

- [ ] **Step 5: Update sources/__init__.py to export VectorSource**

- [ ] **Step 6: Run the tests**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_sources.py::TestVectorSource -v`
Expected: PASS

- [ ] **Step 7: Refactor and verify full suite**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add src/brains/sources/vector_source.py src/brains/data/knowledge.json src/brains/sources/__init__.py tests/test_sources.py
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: vector similarity data source with knowledge fixtures"
```

---

### Task 5: Graph data source

**Files:**
- Create: `src/brains/sources/graph_source.py`
- Create: `src/brains/data/graph.json`
- Modify: `src/brains/sources/__init__.py`
- Modify: `tests/test_sources.py`

- [ ] **Step 1: Write failing test for the graph source**

```python
# In tests/test_sources.py, add:
from brains.sources.graph_source import GraphSource


class TestGraphSource:
    @pytest.fixture
    def source(self):
        return GraphSource()

    def test_implements_protocol(self, source):
        assert isinstance(source, DataSource)

    def test_describe_returns_schema(self, source):
        schema = source.describe()
        assert schema.name == "graph"
        assert len(schema.capabilities) > 0

    def test_query_neighbors(self, source):
        results = source.query({"operation": "neighbors", "node": "Python"})
        assert len(results) > 0
        for r in results:
            assert r.source == "graph"
            assert "node" in r.data or "neighbor" in r.data

    def test_query_path(self, source):
        results = source.query({"operation": "path", "from": "Python", "to": "OpenAI"})
        # May or may not find a path depending on graph structure
        for r in results:
            assert r.source == "graph"

    def test_query_unknown_node_returns_empty(self, source):
        results = source.query({"operation": "neighbors", "node": "NonexistentThing"})
        assert results == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_sources.py::TestGraphSource -v`
Expected: FAIL

- [ ] **Step 3: Create the graph fixture data**

```json
// src/brains/data/graph.json
{
    "nodes": [
        {"id": "Python", "type": "language", "description": "General-purpose programming language"},
        {"id": "PyTorch", "type": "framework", "description": "Deep learning framework"},
        {"id": "TensorFlow", "type": "framework", "description": "Machine learning framework by Google"},
        {"id": "Transformers", "type": "architecture", "description": "Attention-based neural architecture"},
        {"id": "GPT", "type": "model_family", "description": "Generative Pre-trained Transformer series"},
        {"id": "BERT", "type": "model_family", "description": "Bidirectional Encoder Representations from Transformers"},
        {"id": "OpenAI", "type": "company", "description": "AI research company"},
        {"id": "Google", "type": "company", "description": "Technology company"},
        {"id": "Meta", "type": "company", "description": "Technology and social media company"},
        {"id": "Anthropic", "type": "company", "description": "AI safety company"},
        {"id": "Claude", "type": "model_family", "description": "Anthropic's AI assistant models"},
        {"id": "LLaMA", "type": "model_family", "description": "Meta's open-source large language models"},
        {"id": "Attention", "type": "concept", "description": "Mechanism for weighing input relevance"},
        {"id": "Embeddings", "type": "concept", "description": "Dense vector representations of data"},
        {"id": "RAG", "type": "technique", "description": "Retrieval-Augmented Generation"},
        {"id": "Vector Database", "type": "technology", "description": "Database optimized for similarity search"},
        {"id": "Fine-tuning", "type": "technique", "description": "Adapting a pre-trained model to specific tasks"},
        {"id": "Rust", "type": "language", "description": "Systems programming language"},
        {"id": "JavaScript", "type": "language", "description": "Web programming language"},
        {"id": "CUDA", "type": "technology", "description": "NVIDIA GPU computing platform"}
    ],
    "edges": [
        {"from": "PyTorch", "to": "Python", "relation": "implemented_in"},
        {"from": "TensorFlow", "to": "Python", "relation": "implemented_in"},
        {"from": "GPT", "to": "Transformers", "relation": "based_on"},
        {"from": "BERT", "to": "Transformers", "relation": "based_on"},
        {"from": "Claude", "to": "Transformers", "relation": "based_on"},
        {"from": "LLaMA", "to": "Transformers", "relation": "based_on"},
        {"from": "OpenAI", "to": "GPT", "relation": "created"},
        {"from": "Google", "to": "BERT", "relation": "created"},
        {"from": "Google", "to": "TensorFlow", "relation": "created"},
        {"from": "Google", "to": "Transformers", "relation": "created"},
        {"from": "Meta", "to": "LLaMA", "relation": "created"},
        {"from": "Meta", "to": "PyTorch", "relation": "created"},
        {"from": "Anthropic", "to": "Claude", "relation": "created"},
        {"from": "Transformers", "to": "Attention", "relation": "uses"},
        {"from": "Embeddings", "to": "Vector Database", "relation": "stored_in"},
        {"from": "RAG", "to": "Vector Database", "relation": "uses"},
        {"from": "RAG", "to": "Embeddings", "relation": "uses"},
        {"from": "Fine-tuning", "to": "GPT", "relation": "applied_to"},
        {"from": "Fine-tuning", "to": "LLaMA", "relation": "applied_to"},
        {"from": "PyTorch", "to": "CUDA", "relation": "uses"},
        {"from": "TensorFlow", "to": "CUDA", "relation": "uses"},
        {"from": "OpenAI", "to": "Python", "relation": "uses"},
        {"from": "Anthropic", "to": "Python", "relation": "uses"},
        {"from": "Anthropic", "to": "Rust", "relation": "uses"}
    ]
}
```

- [ ] **Step 4: Create the graph source**

```python
# src/brains/sources/graph_source.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from brains.sources.base import DataResult, DataSourceSchema

DATA_DIR = Path(__file__).parent.parent / "data"


class GraphSource:
    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._node_attrs: dict[str, dict] = {}
        self._load_fixture_data()

    def _load_fixture_data(self) -> None:
        fixture_path = DATA_DIR / "graph.json"
        if not fixture_path.exists():
            return

        with open(fixture_path) as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            node_id = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            self._graph.add_node(node_id, **attrs)
            self._node_attrs[node_id] = attrs

        for edge in data.get("edges", []):
            self._graph.add_edge(
                edge["from"], edge["to"],
                relation=edge.get("relation", "related_to"),
            )

    def query(self, params: dict[str, Any]) -> list[DataResult]:
        operation = params.get("operation", "neighbors")

        if operation == "neighbors":
            return self._query_neighbors(params)
        elif operation == "path":
            return self._query_path(params)
        elif operation == "subgraph":
            return self._query_subgraph(params)
        return []

    def _query_neighbors(self, params: dict[str, Any]) -> list[DataResult]:
        node = params.get("node", "")
        if node not in self._graph:
            return []

        results = []
        # Outgoing edges
        for _, neighbor, edge_data in self._graph.out_edges(node, data=True):
            results.append(DataResult(
                source="graph",
                data={
                    "neighbor": neighbor,
                    "relation": edge_data.get("relation", "related_to"),
                    "direction": "outgoing",
                    **self._node_attrs.get(neighbor, {}),
                },
                score=1.0,
            ))
        # Incoming edges
        for predecessor, _, edge_data in self._graph.in_edges(node, data=True):
            results.append(DataResult(
                source="graph",
                data={
                    "neighbor": predecessor,
                    "relation": edge_data.get("relation", "related_to"),
                    "direction": "incoming",
                    **self._node_attrs.get(predecessor, {}),
                },
                score=1.0,
            ))
        return results

    def _query_path(self, params: dict[str, Any]) -> list[DataResult]:
        source_node = params.get("from", "")
        target_node = params.get("to", "")

        if source_node not in self._graph or target_node not in self._graph:
            return []

        try:
            # Use undirected view for path finding
            path = nx.shortest_path(self._graph.to_undirected(), source_node, target_node)
        except nx.NetworkXNoPath:
            return []

        return [DataResult(
            source="graph",
            data={
                "path": path,
                "length": len(path) - 1,
                "from": source_node,
                "to": target_node,
            },
            score=1.0 / len(path),
        )]

    def _query_subgraph(self, params: dict[str, Any]) -> list[DataResult]:
        node = params.get("node", "")
        depth = params.get("depth", 1)

        if node not in self._graph:
            return []

        # BFS to collect nodes within depth
        visited = {node}
        frontier = [node]
        for _ in range(depth):
            next_frontier = []
            for n in frontier:
                for neighbor in set(self._graph.successors(n)) | set(self._graph.predecessors(n)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        return [
            DataResult(
                source="graph",
                data={
                    "node": n,
                    **self._node_attrs.get(n, {}),
                },
                score=1.0,
            )
            for n in visited
        ]

    def describe(self) -> DataSourceSchema:
        return DataSourceSchema(
            name="graph",
            description=f"Knowledge graph of technology relationships with {self._graph.number_of_nodes()} nodes and {self._graph.number_of_edges()} edges. Supports neighbor lookup, path finding, and subgraph extraction.",
            capabilities=["Neighbor lookup", "Shortest path", "Subgraph extraction"],
            sample_queries=[
                '{"operation": "neighbors", "node": "Python"}',
                '{"operation": "path", "from": "PyTorch", "to": "Google"}',
                '{"operation": "subgraph", "node": "Transformers", "depth": 2}',
            ],
        )
```

- [ ] **Step 5: Update sources/__init__.py**

- [ ] **Step 6: Run the tests**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_sources.py::TestGraphSource -v`
Expected: PASS

- [ ] **Step 7: Refactor and verify full suite**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add src/brains/sources/graph_source.py src/brains/data/graph.json src/brains/sources/__init__.py tests/test_sources.py
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: graph data source with networkx and technology relationship fixtures"
```

---

### Task 6: Query engine (the core orchestrator)

**Files:**
- Create: `src/brains/query_engine.py`
- Create: `tests/test_query_engine.py`

- [ ] **Step 1: Write failing test for the query engine**

```python
# tests/test_query_engine.py
import pytest
from brains.query_engine import QueryEngine
from brains.llm.fake_provider import FakeLLMProvider
from brains.sources.sql_source import SQLiteSource
from brains.sources.vector_source import VectorSource
from brains.sources.graph_source import GraphSource
from brains.models import QueryRequest


class TestQueryEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        provider = FakeLLMProvider()
        sources = {
            "sql": SQLiteSource(db_path=tmp_path / "test.db"),
            "vector": VectorSource(),
            "graph": GraphSource(),
        }
        return QueryEngine(
            llm=provider,
            sources=sources,
            model="gpt-4o-mini",
            default_context_budget=4000,
        )

    def test_query_returns_response(self, engine):
        request = QueryRequest(query="What AI companies were founded after 2015?")
        response = engine.execute(request)
        assert response.sources_consulted
        assert response.metadata.total_results >= 0
        assert response.metadata.interpretation_model == "fake"

    def test_query_with_source_filter(self, engine):
        request = QueryRequest(
            query="What AI companies exist?",
            sources=["sql"],
        )
        response = engine.execute(request)
        for result in response.results:
            assert result.source == "sql"

    def test_query_with_raw_format_skips_synthesis(self, engine):
        request = QueryRequest(
            query="List all companies",
            response_format="raw",
        )
        response = engine.execute(request)
        assert response.metadata.synthesis_model is None

    def test_query_with_narrative_format_synthesizes(self, engine):
        request = QueryRequest(
            query="Tell me about AI companies",
            response_format="narrative",
        )
        response = engine.execute(request)
        assert response.answer is not None
        assert response.metadata.synthesis_model is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_query_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create the query engine**

```python
# src/brains/query_engine.py
from __future__ import annotations

import json
import logging
from typing import Any

from brains.llm.provider import LLMProvider
from brains.models import QueryMetadata, QueryRequest, QueryResponse, SourceResult
from brains.sources.base import DataSource, DataResult

logger = logging.getLogger(__name__)


INTERPRETATION_SYSTEM_PROMPT = """You are a query planner for a data service. Given a user query, decide which data sources to consult and what queries to run.

Available data sources:
{source_descriptions}

Return a JSON object with a "queries" array. Each query object must have:
- "source": the data source name (one of: {source_names})
- "query": the specific query to run (SQL for sql source, natural text for vector source, JSON operation for graph source)
- "reasoning": why this source and query are appropriate

Route queries to the most relevant sources. Use multiple sources when the question benefits from cross-referencing."""


SYNTHESIS_SYSTEM_PROMPT = """You are a data synthesis assistant. Synthesize the following query results into a clear, accurate response. Include source attribution.

Return a JSON object with:
- "answer": a clear text answer synthesizing all results
- "confidence": overall confidence score between 0 and 1"""


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

    def execute(self, request: QueryRequest) -> QueryResponse:
        context_budget = request.context_budget or self._default_context_budget

        # Step 1: Interpret the query into a plan
        query_plan = self._interpret(request)

        # Step 2: Execute queries against selected sources
        all_results: list[SourceResult] = []
        sources_consulted: list[str] = []
        total_tokens = 0

        for planned_query in query_plan:
            source_name = planned_query["source"]

            # Apply source filter if specified
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

        # Step 3: Optionally synthesize results
        answer = None
        synthesis_model = None

        if request.response_format != "raw" and all_results:
            synthesis = self._synthesize(request.query, all_results, context_budget)
            answer = synthesis.get("answer")
            synthesis_model = self._model

        return QueryResponse(
            answer=answer,
            sources_consulted=sources_consulted,
            results=all_results,
            metadata=QueryMetadata(
                total_results=sum(len(r.data) for r in all_results),
                context_tokens_used=total_tokens,
                interpretation_model=self._model,
                synthesis_model=synthesis_model,
            ),
        )

    def _interpret(self, request: QueryRequest) -> list[dict[str, Any]]:
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

        try:
            plan = json.loads(response.content)
            return plan.get("queries", [])
        except (json.JSONDecodeError, AttributeError):
            logger.error("Failed to parse query plan from LLM response")
            return [{"source": "sql", "query": request.query, "reasoning": "fallback"}]

    def _build_query_params(self, source_name: str, query: str | dict) -> dict[str, Any]:
        if source_name == "sql":
            return {"sql": query if isinstance(query, str) else str(query)}
        elif source_name == "vector":
            return {"text": query if isinstance(query, str) else str(query), "top_k": 5}
        elif source_name == "graph":
            if isinstance(query, dict):
                return query
            # Parse JSON string query for graph
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
    ) -> dict[str, Any]:
        results_text = json.dumps(
            [{"source": r.source, "data": r.data[:10]} for r in results],
            indent=2,
            default=str,
        )

        # Truncate results to fit context budget (rough estimate: 4 chars per token)
        max_chars = context_budget * 4
        if len(results_text) > max_chars:
            results_text = results_text[:max_chars] + "\n... (truncated)"

        response = self._llm.complete(
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original query: {original_query}\n\nResults:\n{results_text}"},
            ],
            model=self._model,
            temperature=0.0,
            response_format="json",
        )

        try:
            return json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            return {"answer": response.content, "confidence": 0.5}

    def _estimate_tokens(self, result: SourceResult) -> int:
        text = json.dumps(result.data, default=str)
        return len(text) // 4
```

- [ ] **Step 4: Run the tests**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_query_engine.py -v`
Expected: PASS

- [ ] **Step 5: Refactor and verify full suite**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add src/brains/query_engine.py tests/test_query_engine.py
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: query engine with LLM-driven interpretation and synthesis"
```

---

### Task 7: FastAPI application and HTTP endpoints

**Files:**
- Create: `src/brains/main.py`
- Create: `tests/conftest.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing test for HTTP endpoints**

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Test client with fake LLM provider."""
    import os
    os.environ["BRAINS_LLM_PROVIDER"] = "fake"
    os.environ["BRAINS_OPENAI_API_KEY"] = "not-needed"

    from brains.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


# tests/test_api.py
import pytest


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["sources_loaded"] == 3
        assert data["llm_provider"] == "fake"


class TestSourcesEndpoint:
    def test_sources_lists_all_sources(self, client):
        response = client.get("/sources")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        names = {s["name"] for s in data["sources"]}
        assert names == {"sql", "vector", "graph"}


class TestQueryEndpoint:
    def test_query_returns_structured_response(self, client):
        response = client.post("/query", json={
            "query": "What AI companies were founded after 2015?",
        })
        assert response.status_code == 200
        data = response.json()
        assert "sources_consulted" in data
        assert "results" in data
        assert "metadata" in data

    def test_query_with_source_filter(self, client):
        response = client.post("/query", json={
            "query": "List companies",
            "sources": ["sql"],
        })
        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            assert result["source"] == "sql"

    def test_query_with_raw_format(self, client):
        response = client.post("/query", json={
            "query": "List companies",
            "response_format": "raw",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["synthesis_model"] is None

    def test_query_empty_returns_422(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_with_context_budget(self, client):
        response = client.post("/query", json={
            "query": "Tell me about AI companies",
            "context_budget": 1000,
        })
        assert response.status_code == 200
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_api.py -v`
Expected: FAIL — main module not found or create_app not defined

- [ ] **Step 3: Create the FastAPI application**

```python
# src/brains/main.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from brains.config import Settings
from brains.llm import create_provider
from brains.models import (
    DataSourceInfo,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourcesResponse,
)
from brains.query_engine import QueryEngine
from brains.sources.graph_source import GraphSource
from brains.sources.sql_source import SQLiteSource
from brains.sources.vector_source import VectorSource

logger = logging.getLogger(__name__)

_engine: QueryEngine | None = None
_settings: Settings | None = None


def _init_engine(settings: Settings) -> QueryEngine:
    llm = create_provider(settings.llm_provider, api_key=settings.openai_api_key)

    sources = {
        "sql": SQLiteSource(),
        "vector": VectorSource(),
        "graph": GraphSource(),
    }

    return QueryEngine(
        llm=llm,
        sources=sources,
        model=settings.llm_model,
        default_context_budget=settings.default_context_budget,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _settings
    _settings = Settings()
    _engine = _init_engine(_settings)
    logger.info(
        "brAIns started: provider=%s, model=%s, sources=%d",
        _settings.llm_provider,
        _settings.llm_model,
        len(_engine._sources),
    )
    yield
    _engine = None
    _settings = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="brAIns",
        description="AI-augmented curated data service. Query specialized, curated data sources through an LLM-powered interface.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            sources_loaded=len(_engine._sources) if _engine else 0,
            llm_provider=_settings.llm_provider if _settings else "unknown",
        )

    @app.get("/sources", response_model=SourcesResponse)
    async def sources():
        if not _engine:
            return SourcesResponse(sources=[])

        source_infos = []
        for name, source in _engine._sources.items():
            schema = source.describe()
            source_infos.append(DataSourceInfo(
                name=schema.name,
                description=schema.description,
                capabilities=schema.capabilities,
                sample_queries=schema.sample_queries,
            ))
        return SourcesResponse(sources=source_infos)

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        return _engine.execute(request)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = Settings()
    uvicorn.run("brains.main:app", host=settings.host, port=settings.port, reload=True)
```

- [ ] **Step 4: Run the tests**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Refactor and verify full suite**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add src/brains/main.py tests/conftest.py tests/test_api.py
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: FastAPI application with health, sources, and query endpoints"
```

---

### Task 8: Docker and docker-compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `tests/test_smoke.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the container smoke test**

```python
# tests/test_smoke.py
"""Container smoke tests.

Run with: BRAINS_SMOKE_TEST=1 pytest tests/test_smoke.py -v
Requires docker-compose up -d to be running.
"""
import os
import pytest
import httpx

BRAINS_URL = os.environ.get("BRAINS_URL", "http://localhost:8000")
pytestmark = pytest.mark.skipif(
    os.environ.get("BRAINS_SMOKE_TEST") != "1",
    reason="Container smoke tests require BRAINS_SMOKE_TEST=1",
)


class TestContainerSmoke:
    def test_health_endpoint(self):
        response = httpx.get(f"{BRAINS_URL}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["sources_loaded"] == 3

    def test_sources_endpoint(self):
        response = httpx.get(f"{BRAINS_URL}/sources", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3

    def test_query_endpoint(self):
        response = httpx.post(
            f"{BRAINS_URL}/query",
            json={"query": "What AI companies exist?"},
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "metadata" in data

    def test_openapi_docs_available(self):
        response = httpx.get(f"{BRAINS_URL}/openapi.json", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "brAIns"
```

- [ ] **Step 2: Create the Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); assert r.status_code == 200"

CMD ["uvicorn", "brains.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  brains:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BRAINS_LLM_PROVIDER=${BRAINS_LLM_PROVIDER:-fake}
      - BRAINS_LLM_MODEL=${BRAINS_LLM_MODEL:-gpt-4o-mini}
      - BRAINS_OPENAI_API_KEY=${BRAINS_OPENAI_API_KEY:-}
      - BRAINS_DEFAULT_CONTEXT_BUDGET=${BRAINS_DEFAULT_CONTEXT_BUDGET:-4000}
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; r = httpx.get('http://localhost:8000/health'); assert r.status_code == 200"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
```

- [ ] **Step 4: Update .gitignore**

Add:
```
__pycache__/
*.pyc
*.egg-info/
.venv/
dist/
build/
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 5: Build and run the container**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && docker compose build`
Expected: Build succeeds

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && docker compose up -d && sleep 5 && docker compose ps`
Expected: Service is running and healthy

- [ ] **Step 6: Run the smoke tests**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && BRAINS_SMOKE_TEST=1 python3 -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 7: Clean up containers**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && docker compose down`

- [ ] **Step 8: Refactor and verify full suite**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS (smoke tests skipped without env var)

- [ ] **Step 9: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add Dockerfile docker-compose.yml .gitignore tests/test_smoke.py
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "feat: Docker and docker-compose with container smoke tests"
```

---

### Task 9: README and final integration verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
# brAIns

AI-augmented curated data service. Agents query specialized data sources through an LLM-powered interface.

## Quick start

```bash
# With fake LLM (no API key needed)
docker compose up -d

# With OpenAI
BRAINS_OPENAI_API_KEY=sk-... BRAINS_LLM_PROVIDER=openai docker compose up -d
```

## API

- `GET /health` — Health check
- `GET /sources` — List available data sources
- `POST /query` — Query data sources

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What AI companies were founded after 2015?"}'
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINS_LLM_PROVIDER` | `openai` | LLM provider (`openai`, `fake`) |
| `BRAINS_LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `BRAINS_OPENAI_API_KEY` | | OpenAI API key |
| `BRAINS_DEFAULT_CONTEXT_BUDGET` | `4000` | Max tokens in response |

## Data sources

- **sql** — SQLite with tech company data
- **vector** — In-memory semantic search over AI/ML knowledge
- **graph** — NetworkX knowledge graph of technology relationships
```

- [ ] **Step 2: Run the full test suite one final time**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 3: Build and smoke test containers one final time**

Run: `cd /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp && docker compose build && docker compose up -d && sleep 5 && BRAINS_SMOKE_TEST=1 python3 -m pytest tests/test_smoke.py -v && docker compose down`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp add README.md
git -C /Volumes/RayCue-Drive/Documents/projects/brAIns/.worktrees/feature/brains-mvp commit -m "docs: add README with quick start and API documentation"
```

---

## Remember

- Every test must pass for legitimate reasons. Do not weaken, delete, or bypass a valid test to make it pass.
- Run the full test suite after every task, not just the new tests.
- Container smoke tests require `BRAINS_SMOKE_TEST=1` and a running docker-compose stack.
- The fake LLM provider must produce deterministic, structurally valid responses — it is the foundation for all integration tests.
- Keep data fixtures small but representative. They demonstrate the pattern, not production scale.
- When the query engine's LLM interpretation returns an invalid plan, fall back gracefully to the SQL source rather than erroring.
- The `DataSource` protocol is the primary extension point. Adding a new data source should require: (1) implement the protocol, (2) register in `main.py`, (3) add fixture data. No other files should need to change.
