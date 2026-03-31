import pytest

from brains.llm.fake_provider import FakeLLMProvider
from brains.models import QueryRequest
from brains.query_engine import QueryEngine
from brains.sources.graph_source import GraphSource
from brains.sources.sql_source import SQLiteSource
from brains.sources.vector_source import VectorSource


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
