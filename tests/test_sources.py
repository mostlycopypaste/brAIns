import pytest

from brains.sources.base import DataSource
from brains.sources.sql_source import SQLiteSource
from brains.sources.vector_source import VectorSource
from brains.sources.graph_source import GraphSource


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
        assert len(schema.sample_queries) > 0

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
        assert len(results) == 0

    def test_query_delete_rejected(self, source):
        results = source.query({"sql": "DELETE FROM companies"})
        assert len(results) == 0
        # Table should still be intact
        results = source.query({"sql": "SELECT COUNT(*) as cnt FROM companies"})
        assert len(results) > 0
        assert results[0].data["cnt"] > 0


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
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_empty_text_returns_empty(self, source):
        results = source.query({"text": "", "top_k": 3})
        assert results == []


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
            assert "neighbor" in r.data

    def test_query_path(self, source):
        results = source.query({"operation": "path", "from": "Python", "to": "OpenAI"})
        assert len(results) > 0
        for r in results:
            assert r.source == "graph"
            assert "path" in r.data
            assert "length" in r.data
            assert r.data["length"] >= 1

    def test_query_unknown_node_returns_empty(self, source):
        results = source.query({"operation": "neighbors", "node": "NonexistentThing"})
        assert results == []
