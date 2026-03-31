"""Smoke tests for the brAIns HTTP surface.

These exercise the real HTTP endpoints via FastAPI's TestClient
without requiring Docker or external services.
"""


class TestSmoke:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["sources_loaded"] == 3
        assert data["llm_provider"]

    def test_sources_endpoint(self, client):
        response = client.get("/sources")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        names = {s["name"] for s in data["sources"]}
        assert names == {"sql", "vector", "graph"}
        for s in data["sources"]:
            assert s["description"]
            assert len(s["capabilities"]) > 0
            assert len(s["sample_queries"]) > 0

    def test_query_endpoint(self, client):
        response = client.post(
            "/query",
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

    def test_openapi_docs_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "brAIns"
        assert "/health" in data["paths"]
        assert "/sources" in data["paths"]
        assert "/query" in data["paths"]
