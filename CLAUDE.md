# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

brAIns is an AI-augmented data service that routes natural language queries to specialized data sources through an LLM-powered interface. It acts as a query planner and synthesizer: the LLM decides which sources to consult, then combines results into a coherent response.

## Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run locally with fake LLM (no API key needed)
BRAINS_LLM_PROVIDER=fake python -m brains.main

# Run locally with OpenAI
BRAINS_OPENAI_API_KEY=sk-... python -m brains.main

# Run locally with Ollama
BRAINS_LLM_PROVIDER=ollama BRAINS_LLM_MODEL=llama3.2 python -m brains.main

# Run tests
pytest                           # All tests
pytest tests/test_query_engine.py  # Single file
pytest -k test_fake_provider     # Single test by name

# Lint and format
ruff check src tests             # Check for issues
ruff format src tests            # Auto-format code

# Docker
docker compose up -d             # Start service
docker compose logs -f           # Follow logs
```

## Architecture

### Query Flow
1. **Interpretation**: `QueryEngine._interpret()` sends user query + source descriptions to LLM, which returns a plan of which sources to query and how
2. **Execution**: Each planned query runs against its DataSource, returning scored results
3. **Synthesis**: `QueryEngine._synthesize()` sends original query + all results to LLM for a unified response (unless `response_format="raw"`)

### Key Abstractions

**LLMProvider** (`brains/llm/provider.py`): Protocol for pluggable LLM backends
- `OpenAIProvider`: Production LLM using OpenAI API
- `OllamaProvider`: Local LLM using Ollama (OpenAI-compatible API)
- `FakeProvider`: Deterministic responses for testing without API keys

**DataSource** (`brains/sources/base.py`): Protocol for pluggable data sources
- `SQLiteSource`: In-memory company database, takes SQL queries
- `VectorSource`: Semantic search over AI/ML knowledge chunks
- `GraphSource`: NetworkX graph of technology relationships
- Each source returns `DataResult[]` with score and metadata

**Configuration** (`brains/config.py`): pydantic-settings for environment variables
- `BRAINS_LLM_PROVIDER`: `openai` (default), `ollama`, or `fake`
- `BRAINS_OPENAI_API_KEY`: Required when provider=openai
- `BRAINS_OLLAMA_BASE_URL`: Ollama endpoint (default: `http://localhost:11434/v1`)
- `BRAINS_LLM_MODEL`: Model name (default: `gpt-4o-mini`)
- `BRAINS_DEFAULT_CONTEXT_BUDGET`: Max tokens in response (default: 4000)

### FastAPI Lifespan
Sources and LLM are initialized once at startup (`lifespan` context manager in `main.py`), not per-request. Global `_engine` and `_settings` hold the initialized state.

### Response Formats
- `raw`: No LLM synthesis, just return data results
- `structured`: LLM returns JSON-formatted answer
- `narrative`: LLM returns prose summary

## Extending

### Adding a Data Source
1. Implement the `DataSource` protocol (requires `query()` and `describe()` methods)
2. Register in `main.py:_init_engine()`
3. Update `query_engine.py:_build_query_params()` if source has unique query format

### Adding an LLM Provider
1. Implement the `LLMProvider` protocol (requires `complete()` method returning `LLMResponse`)
2. Add provider factory logic to `llm/__init__.py:create_provider()`

## Testing Strategy
- Use `FakeProvider` for deterministic LLM responses in tests
- Integration tests (`test_api.py`) use FastAPI TestClient
- Unit tests mock individual components
- `conftest.py` provides shared fixtures (fake LLM, test client, sample data)
