"""Smoke tests for the brAIns HTTP surface.

These run against a live Docker container via httpx.
Collected only when BRAINS_SMOKE_TEST=1 (see conftest.py collect_ignore).
"""

import os

import httpx

BRAINS_URL = os.environ.get("BRAINS_URL", "http://localhost:8000")


class TestSmoke:
    def test_health_endpoint(self):
        response = httpx.get(f"{BRAINS_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["sources_loaded"] == 3
        assert data["llm_provider"]

    def test_sources_endpoint(self):
        response = httpx.get(f"{BRAINS_URL}/sources")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        names = {s["name"] for s in data["sources"]}
        assert names == {"sql", "vector", "graph"}
        for s in data["sources"]:
            assert s["description"]
            assert len(s["capabilities"]) > 0
            assert len(s["sample_queries"]) > 0

    def test_query_endpoint(self):
        response = httpx.post(
            f"{BRAINS_URL}/query",
            json={"query": "What AI companies exist?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "metadata" in data
        assert len(data["sources_consulted"]) > 0
        for result in data["results"]:
            assert "source" in result
            assert "confidence" in result
            assert "data" in result
            assert "query_used" in result

    def test_openapi_docs_available(self):
        response = httpx.get(f"{BRAINS_URL}/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "brAIns"
        assert "/health" in data["paths"]
        assert "/sources" in data["paths"]
        assert "/query" in data["paths"]
