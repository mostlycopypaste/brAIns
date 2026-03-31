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

- `GET /health` -- Health check
- `GET /sources` -- List available data sources
- `POST /query` -- Query data sources

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

- **sql** -- SQLite with tech company data
- **vector** -- In-memory semantic search over AI/ML knowledge
- **graph** -- NetworkX knowledge graph of technology relationships
