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
