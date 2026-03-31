# brAIns MVP Test Plan

**Date:** 2026-03-30
**Implementation plan:** `docs/plans/2026-03-30-brains-mvp.md`

**Testing strategy allocation:**
- 50% container smoke + integration tests (highest priority)
- 30% LLM abstraction boundary tests
- 20% unit tests for routing/transformation logic

**Constraints:** Single command runnable (`pytest`), zero external dependencies (fake LLM provider, no OpenAI key needed for CI).

---

## Harnesses

### H1: FastAPI TestClient (in-process)
- Uses `fastapi.testclient.TestClient` with fake LLM provider injected via env vars
- Exercises the full HTTP surface including serialization, validation, and Pydantic models
- No container required; runs in `pytest` without Docker
- Defined in `tests/conftest.py` as the `client` fixture

### H2: Container smoke (docker-compose)
- Uses `httpx` against `http://localhost:8000`
- Requires `docker compose up -d` with `BRAINS_LLM_PROVIDER=fake`
- Gated behind `BRAINS_SMOKE_TEST=1` env var; skipped otherwise
- Validates the container builds, starts, passes healthcheck, and serves all endpoints

### H3: Direct unit (pytest, no HTTP)
- Instantiates components directly (QueryEngine, data sources, LLM providers)
- Uses `tmp_path` fixture for SQLite isolation
- No HTTP, no app lifecycle

---

## Tests (ordered by priority)

### 1. Container health check responds after startup

- **Name:** Container responds to health check after docker-compose up
- **Type:** scenario
- **Disposition:** new
- **Harness:** H2
- **Preconditions:** `docker compose up -d` with `BRAINS_LLM_PROVIDER=fake`, container healthy
- **Actions:** `GET /health`
- **Expected outcome:** HTTP 200; body has `status: "ok"`, `sources_loaded: 3`, `llm_provider` is a non-empty string
- **Interactions:** Docker build, pip install inside container, FastAPI lifespan startup, all three data source fixture loads

### 2. Container lists all data sources

- **Name:** Container returns all three data sources from /sources
- **Type:** scenario
- **Disposition:** new
- **Harness:** H2
- **Preconditions:** Container running and healthy
- **Actions:** `GET /sources`
- **Expected outcome:** HTTP 200; body has `sources` array of length 3; names are `{"sql", "vector", "graph"}`; each source has non-empty `description`, `capabilities`, and `sample_queries`
- **Interactions:** All three data source `describe()` methods

### 3. Container handles full query pipeline

- **Name:** Container processes a query end-to-end and returns structured response
- **Type:** scenario
- **Disposition:** new
- **Harness:** H2
- **Preconditions:** Container running and healthy
- **Actions:** `POST /query` with `{"query": "What AI companies were founded after 2015?"}`
- **Expected outcome:** HTTP 200; body has `sources_consulted` (non-empty list), `results` (list of objects with `source`, `confidence`, `data`, `query_used`), `metadata` with `total_results >= 0` and `interpretation_model` present
- **Interactions:** Fake LLM interpretation, fake LLM synthesis, SQL data source query, Pydantic serialization

### 4. Container serves OpenAPI spec

- **Name:** OpenAPI JSON schema is accessible for agent consumers
- **Type:** scenario
- **Disposition:** new
- **Harness:** H2
- **Preconditions:** Container running and healthy
- **Actions:** `GET /openapi.json`
- **Expected outcome:** HTTP 200; body is valid JSON with `info.title == "brAIns"`; paths include `/health`, `/sources`, `/query`
- **Interactions:** FastAPI automatic OpenAPI generation

### 5. Query endpoint returns structured response via TestClient

- **Name:** POST /query returns well-formed structured response
- **Type:** integration
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized with fake LLM provider
- **Actions:** `POST /query` with `{"query": "What AI companies were founded after 2015?"}`
- **Expected outcome:** HTTP 200; response body matches `QueryResponse` schema: `sources_consulted` present, `results` is a list, `metadata` has `total_results`, `context_tokens_used`, `interpretation_model`
- **Interactions:** Full pipeline: request parsing, LLM interpretation, source routing, source queries, LLM synthesis, response serialization

### 6. Query with source filter restricts results

- **Name:** Source filter limits which data sources are consulted
- **Type:** integration
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized with fake LLM provider
- **Actions:** `POST /query` with `{"query": "List companies", "sources": ["sql"]}`
- **Expected outcome:** HTTP 200; every item in `results` has `source == "sql"`; no other sources appear in `sources_consulted`
- **Interactions:** Fake LLM planner (may suggest multiple sources), source filter logic in QueryEngine, SQL source query

### 7. Raw response format skips synthesis

- **Name:** Setting response_format to "raw" bypasses LLM synthesis step
- **Type:** integration
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized with fake LLM provider
- **Actions:** `POST /query` with `{"query": "List companies", "response_format": "raw"}`
- **Expected outcome:** HTTP 200; `metadata.synthesis_model` is `null`; `answer` is `null`
- **Interactions:** Fake LLM interpretation only (no synthesis call), source queries

### 8. Narrative response format produces synthesized answer

- **Name:** Narrative format triggers LLM synthesis and returns an answer
- **Type:** integration
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized with fake LLM provider
- **Actions:** `POST /query` with `{"query": "Tell me about AI companies", "response_format": "narrative"}`
- **Expected outcome:** HTTP 200; `answer` is a non-null, non-empty string; `metadata.synthesis_model` is not null
- **Interactions:** Fake LLM interpretation + synthesis, source queries

### 9. Context budget is respected

- **Name:** Context budget parameter limits response size
- **Type:** integration
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized with fake LLM provider
- **Actions:** `POST /query` with `{"query": "Tell me about AI companies", "context_budget": 1000}`
- **Expected outcome:** HTTP 200; `metadata.context_tokens_used` is reported; response completes without error
- **Interactions:** QueryEngine truncation logic, synthesis with truncated context

### 10. Missing query field returns 422

- **Name:** Omitting required query field produces validation error
- **Type:** boundary
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized
- **Actions:** `POST /query` with `{}`
- **Expected outcome:** HTTP 422; response body contains validation error detail for `query` field
- **Interactions:** Pydantic validation, FastAPI error handling

### 11. Health endpoint returns correct state

- **Name:** Health endpoint reports service status and source count
- **Type:** integration
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized with fake LLM provider
- **Actions:** `GET /health`
- **Expected outcome:** HTTP 200; `status == "ok"`, `sources_loaded == 3`, `llm_provider == "fake"`
- **Interactions:** App lifespan globals

### 12. Sources endpoint lists all three sources with metadata

- **Name:** Sources endpoint returns complete metadata for all data sources
- **Type:** integration
- **Disposition:** new
- **Harness:** H1
- **Preconditions:** App initialized with fake LLM provider
- **Actions:** `GET /sources`
- **Expected outcome:** HTTP 200; `sources` has 3 items; names are `{"sql", "vector", "graph"}`; each has non-empty `description`, `capabilities` (list), and `sample_queries` (list)
- **Interactions:** All three data source `describe()` methods, Pydantic serialization

### 13. Fake LLM provider implements protocol

- **Name:** FakeLLMProvider satisfies the LLMProvider protocol at runtime
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Instantiate `FakeLLMProvider()`, check `isinstance(provider, LLMProvider)`
- **Expected outcome:** Returns `True`
- **Interactions:** `runtime_checkable` Protocol

### 14. Fake LLM returns canned response

- **Name:** FakeLLMProvider returns a valid LLMResponse for any input
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Call `provider.complete(messages=[{"role": "user", "content": "test"}], model="gpt-4o-mini")`
- **Expected outcome:** Returns `LLMResponse` with non-null `content`, `model == "fake"`, integer `prompt_tokens` and `completion_tokens`
- **Interactions:** None

### 15. Fake LLM returns structured QueryPlan for planner prompt

- **Name:** FakeLLMProvider returns parseable JSON query plan when system prompt contains "query planner"
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Call `provider.complete()` with system message containing "query planner" and user message about companies; parse `response.content` as JSON
- **Expected outcome:** Parsed JSON has `queries` key containing a non-empty list; each item has `source`, `query`, and `reasoning` keys
- **Interactions:** Fake provider keyword routing logic

### 16. Fake LLM routes company queries to SQL source

- **Name:** FakeLLMProvider routes company-related keywords to the SQL source
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Call fake provider with planner system prompt and user query "What companies were founded after 2015?"
- **Expected outcome:** At least one planned query has `source == "sql"`
- **Interactions:** Fake provider keyword matching

### 17. Fake LLM routes conceptual queries to vector source

- **Name:** FakeLLMProvider routes conceptual/knowledge queries to the vector source
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Call fake provider with planner system prompt and user query "What is machine learning?"
- **Expected outcome:** At least one planned query has `source == "vector"`
- **Interactions:** Fake provider keyword matching

### 18. Fake LLM routes relationship queries to graph source

- **Name:** FakeLLMProvider routes relationship queries to the graph source
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Call fake provider with planner system prompt and user query "What is related to Python?"
- **Expected outcome:** At least one planned query has `source == "graph"`
- **Interactions:** Fake provider keyword matching

### 19. Fake LLM synthesis response is valid JSON

- **Name:** FakeLLMProvider returns valid synthesis JSON when system prompt contains "synthesize"
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Call fake provider with system message containing "synthesize" and user message with results context
- **Expected outcome:** Parsed JSON has `answer` (string) and `confidence` (float between 0 and 1)
- **Interactions:** None

### 20. LLM provider factory creates correct provider

- **Name:** create_provider returns the correct implementation based on provider name
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** Call `create_provider("fake")` and `create_provider("openai", api_key="test-key")`; call `create_provider("unknown")` and `create_provider("openai", api_key="")`
- **Expected outcome:** "fake" returns `FakeLLMProvider`; "openai" with key returns `OpenAIProvider`; "unknown" raises `ValueError`; "openai" without key raises `ValueError`
- **Interactions:** Factory function, import of both provider classes

### 21. QueryEngine routes to SQL and returns results

- **Name:** QueryEngine interprets a company query, routes to SQL, and returns structured results
- **Type:** integration
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** QueryEngine with fake LLM and all three sources
- **Actions:** `engine.execute(QueryRequest(query="What AI companies were founded after 2015?"))`
- **Expected outcome:** `response.sources_consulted` is non-empty; `response.metadata.interpretation_model == "fake"`; `response.metadata.total_results >= 0`
- **Interactions:** Fake LLM interpretation, SQL source query, result aggregation

### 22. QueryEngine respects source filter

- **Name:** QueryEngine only queries sources in the filter list
- **Type:** integration
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** QueryEngine with fake LLM and all three sources
- **Actions:** `engine.execute(QueryRequest(query="What AI companies exist?", sources=["sql"]))`
- **Expected outcome:** All items in `response.results` have `source == "sql"`
- **Interactions:** Fake LLM interpretation (may plan multi-source), filter logic

### 23. QueryEngine raw format skips synthesis

- **Name:** QueryEngine skips LLM synthesis when response_format is "raw"
- **Type:** integration
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** QueryEngine with fake LLM and all three sources
- **Actions:** `engine.execute(QueryRequest(query="List all companies", response_format="raw"))`
- **Expected outcome:** `response.metadata.synthesis_model is None`; `response.answer is None`
- **Interactions:** Fake LLM interpretation only

### 24. QueryEngine narrative format triggers synthesis

- **Name:** QueryEngine calls LLM synthesis when response_format is "narrative"
- **Type:** integration
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** QueryEngine with fake LLM and all three sources
- **Actions:** `engine.execute(QueryRequest(query="Tell me about AI companies", response_format="narrative"))`
- **Expected outcome:** `response.answer is not None`; `response.metadata.synthesis_model is not None`
- **Interactions:** Fake LLM interpretation + synthesis

### 25. SQLite source implements DataSource protocol

- **Name:** SQLiteSource satisfies the DataSource protocol
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** SQLiteSource instantiated with tmp_path
- **Actions:** `isinstance(source, DataSource)`
- **Expected outcome:** `True`
- **Interactions:** `runtime_checkable` Protocol

### 26. SQLite source loads fixture data and returns results

- **Name:** SQLiteSource loads companies.json and answers SELECT queries
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** SQLiteSource instantiated with tmp_path
- **Actions:** `source.query({"sql": "SELECT * FROM companies LIMIT 3"})`
- **Expected outcome:** Returns 1-3 `DataResult` objects; each has `source == "sql"` and `data` dict with `name` key
- **Interactions:** sqlite3, fixture file read

### 27. SQLite source filters by sector

- **Name:** SQL WHERE clause correctly filters company records
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** SQLiteSource with loaded fixtures
- **Actions:** `source.query({"sql": "SELECT * FROM companies WHERE sector = 'AI'"})`
- **Expected outcome:** All returned results have `data["sector"] == "AI"`; count > 0
- **Interactions:** sqlite3

### 28. SQLite source rejects write operations

- **Name:** Non-SELECT SQL statements are rejected and return empty results
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** SQLiteSource with loaded fixtures
- **Actions:** `source.query({"sql": "DROP TABLE companies"})` and `source.query({"sql": "DELETE FROM companies"})`
- **Expected outcome:** Both return empty list; no error raised; table still intact for subsequent queries
- **Interactions:** Statement type allowlist check

### 29. SQLite source describe returns schema

- **Name:** SQLiteSource.describe() returns valid DataSourceSchema
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** SQLiteSource instantiated
- **Actions:** `source.describe()`
- **Expected outcome:** `schema.name == "sql"`; `"companies"` in `schema.description.lower()`; `len(schema.capabilities) > 0`; `len(schema.sample_queries) > 0`
- **Interactions:** None

### 30. Vector source implements protocol and returns ranked results

- **Name:** VectorSource returns results sorted by descending similarity score
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** VectorSource instantiated (loads knowledge.json fixtures)
- **Actions:** `source.query({"text": "neural networks and deep learning", "top_k": 3})`
- **Expected outcome:** Returns 1-3 `DataResult` objects; `source == "vector"`; each has `0.0 <= score <= 1.0` and `"text"` in `data`; scores are in descending order
- **Interactions:** numpy cosine similarity, fixture embedding load

### 31. Vector source returns empty for empty query

- **Name:** VectorSource returns empty list for blank text query
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** VectorSource instantiated
- **Actions:** `source.query({"text": "", "top_k": 3})`
- **Expected outcome:** Returns empty list
- **Interactions:** None

### 32. Graph source returns neighbors

- **Name:** GraphSource returns neighbor nodes for a known entity
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** GraphSource instantiated (loads graph.json fixtures)
- **Actions:** `source.query({"operation": "neighbors", "node": "Python"})`
- **Expected outcome:** Returns non-empty list; each result has `source == "graph"` and `data` with `"neighbor"` key
- **Interactions:** networkx graph traversal

### 33. Graph source finds shortest path

- **Name:** GraphSource finds a path between two connected nodes
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** GraphSource instantiated
- **Actions:** `source.query({"operation": "path", "from": "Python", "to": "OpenAI"})`
- **Expected outcome:** Returns a result with `data["path"]` as a list and `data["length"]` as an integer >= 1
- **Interactions:** networkx shortest_path on undirected view

### 34. Graph source returns empty for unknown node

- **Name:** GraphSource returns empty list for a non-existent node
- **Type:** boundary
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** GraphSource instantiated
- **Actions:** `source.query({"operation": "neighbors", "node": "NonexistentThing"})`
- **Expected outcome:** Returns empty list; no error raised
- **Interactions:** networkx node lookup

### 35. Configuration loads defaults

- **Name:** Settings loads sensible defaults when no env vars are set
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** None
- **Actions:** `Settings(openai_api_key="test-key")`
- **Expected outcome:** `llm_model == "gpt-4o-mini"`, `llm_provider == "openai"`, `host == "0.0.0.0"`, `port == 8000`, `default_context_budget == 4000`
- **Interactions:** pydantic-settings

### 36. Configuration reads from environment

- **Name:** Settings reads BRAINS_-prefixed environment variables
- **Type:** unit
- **Disposition:** new
- **Harness:** H3
- **Preconditions:** Env vars set: `BRAINS_LLM_MODEL=gpt-4o`, `BRAINS_LLM_PROVIDER=fake`, `BRAINS_DEFAULT_CONTEXT_BUDGET=8000`, `BRAINS_OPENAI_API_KEY=sk-test-123`
- **Actions:** `Settings()`
- **Expected outcome:** `llm_model == "gpt-4o"`, `llm_provider == "fake"`, `default_context_budget == 8000`, `openai_api_key == "sk-test-123"`
- **Interactions:** pydantic-settings, os.environ

---

## Test-to-file mapping

| Test # | Test file | Implementation files exercised |
|--------|-----------|-------------------------------|
| 1-4 | `tests/test_smoke.py` | All (via container) |
| 5-12 | `tests/test_api.py` | `main.py`, `query_engine.py`, all sources, fake LLM |
| 10 | `tests/test_api.py` | `models.py` (Pydantic validation) |
| 13-20 | `tests/test_llm.py` | `llm/provider.py`, `llm/fake_provider.py`, `llm/__init__.py` |
| 21-24 | `tests/test_query_engine.py` | `query_engine.py`, all sources, fake LLM |
| 25-29 | `tests/test_sources.py` | `sources/sql_source.py`, `sources/base.py`, `data/companies.json` |
| 30-31 | `tests/test_sources.py` | `sources/vector_source.py`, `data/knowledge.json` |
| 32-34 | `tests/test_sources.py` | `sources/graph_source.py`, `data/graph.json` |
| 35-36 | `tests/test_config.py` | `config.py` |

## Coverage by strategy allocation

| Category | Tests | Count | % of total |
|----------|-------|-------|------------|
| Container smoke + integration | 1-12, 21-24 | 16 | 44% |
| LLM abstraction boundary | 13-20 | 8 | 22% |
| Unit tests (sources, config, routing) | 25-36 | 12 | 33% |

## Running the tests

```bash
# Unit + integration tests (no Docker needed)
pytest tests/ -v

# Container smoke tests (requires running containers)
docker compose up -d
BRAINS_SMOKE_TEST=1 pytest tests/test_smoke.py -v
docker compose down

# Full suite
docker compose up -d
BRAINS_SMOKE_TEST=1 pytest tests/ -v
docker compose down
```

## Reconciliation notes

1. **Strategy validated.** The implementation plan's `FakeLLMProvider` with keyword-based routing makes the 50% integration allocation feasible without external dependencies. The fake provider's deterministic behavior enables reliable assertions about routing outcomes.

2. **No strategy invalidation found.** The plan's architecture (three endpoints, three sources, two LLM calls per query) maps cleanly to the strategy's emphasis on container smoke + integration tests. The `BRAINS_LLM_PROVIDER=fake` env var mechanism works for both TestClient and container scenarios.

3. **Synthesis step is the critical boundary.** The two-LLM-call pattern (interpret + synthesize) creates a natural test seam. Tests 7-8 and 23-24 specifically verify that the synthesis step is correctly gated by `response_format`.

4. **SQL injection prevention via allowlist is a safety invariant.** Test 28 validates that write operations are rejected. This is the only security-relevant test in the MVP.

5. **OpenAPI auto-generation is an agent-facing contract.** Test 4 validates that AI agent consumers can discover the API programmatically, which is called out as a design goal.
