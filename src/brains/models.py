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
