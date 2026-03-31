from __future__ import annotations

import logging
from contextlib import asynccontextmanager

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
        description=(
            "AI-augmented curated data service. "
            "Query specialized, curated data sources through an LLM-powered interface."
        ),
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
        for source in _engine._sources.values():
            schema = source.describe()
            source_infos.append(
                DataSourceInfo(
                    name=schema.name,
                    description=schema.description,
                    capabilities=schema.capabilities,
                    sample_queries=schema.sample_queries,
                )
            )
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
