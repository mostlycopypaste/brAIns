# brAIns

AI-augmented curated data service. Agents query specialized data sources through an LLM-powered interface.

## Quick start

```bash
# With fake LLM (no API key needed)
docker compose up -d

# With OpenAI
BRAINS_OPENAI_API_KEY=sk-... BRAINS_LLM_PROVIDER=openai docker compose up -d

# With Ollama (local LLM - requires Ollama running on host)
BRAINS_LLM_PROVIDER=ollama BRAINS_LLM_MODEL=llama3.2 docker compose up -d
```

**Note**: When using Ollama with Docker, the container accesses your host's Ollama via `host.docker.internal:11434`. Make sure Ollama is running on your host machine before starting the container.

## API

- `GET /health` -- Health check
- `GET /sources` -- List available data sources
- `POST /query` -- Query data sources

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What AI companies were founded after 2015?"}'
```

## Demo

Start the server with the fake LLM provider (no API key needed):

```bash
pip install -e ".[dev]"
BRAINS_LLM_PROVIDER=fake python -m brains.main
```

### Explore available data

```bash
# Health check
curl http://localhost:8000/health

# List data sources and sample queries
curl http://localhost:8000/sources
```

### Query company data (SQL source)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What companies are in the technology sector?"}'
```

### Query AI concepts (vector search)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain transformer architecture and attention mechanisms"}'
```

### Query technology relationships (graph source)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What technologies are related to Python?"}'
```

### Filter to a specific source

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all companies", "sources": ["sql"]}'
```

### Response formats

```bash
# Raw -- no LLM synthesis, just data
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "companies founded after 2000", "response_format": "raw"}'

# Narrative -- prose summary
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does machine learning relate to neural networks?", "response_format": "narrative"}'
```

### AI agent prompt

To point an AI agent at brAIns:

> You have access to a data service at http://localhost:8000. Use POST /query with a JSON body containing `{"query": "your question"}` to look up information. Available data sources include company profiles (15 tech companies), AI/ML knowledge chunks (concepts like transformers, GANs, reinforcement learning), and a technology relationship graph. Try querying about companies in a specific sector, AI concepts, or how technologies relate to each other. You can also GET /sources to see what's available.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAINS_LLM_PROVIDER` | `openai` | LLM provider (`openai`, `ollama`, `fake`) |
| `BRAINS_LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `BRAINS_OPENAI_API_KEY` | | OpenAI API key (required for `openai` provider) |
| `BRAINS_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint (for `ollama` provider) |
| `BRAINS_DEFAULT_CONTEXT_BUDGET` | `4000` | Max tokens in response |

## Data sources

- **sql** -- SQLite with tech company data
- **vector** -- In-memory semantic search over AI/ML knowledge
- **graph** -- NetworkX knowledge graph of technology relationships
