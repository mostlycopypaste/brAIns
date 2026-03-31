class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["sources_loaded"] == 3
        assert data["llm_provider"] == "fake"


class TestSourcesEndpoint:
    def test_sources_lists_all_sources(self, client):
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


class TestQueryEndpoint:
    def test_query_returns_structured_response(self, client):
        response = client.post(
            "/query",
            json={"query": "What AI companies were founded after 2015?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sources_consulted" in data
        assert "results" in data
        assert "metadata" in data
        assert data["metadata"]["total_results"] >= 0
        assert data["metadata"]["interpretation_model"]

    def test_query_with_source_filter(self, client):
        response = client.post(
            "/query",
            json={"query": "List companies", "sources": ["sql"]},
        )
        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            assert result["source"] == "sql"

    def test_query_with_raw_format(self, client):
        response = client.post(
            "/query",
            json={"query": "List companies", "response_format": "raw"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["synthesis_model"] is None
        assert data["answer"] is None

    def test_query_empty_returns_422(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_with_context_budget(self, client):
        response = client.post(
            "/query",
            json={"query": "Tell me about AI companies", "context_budget": 1000},
        )
        assert response.status_code == 200

    def test_query_narrative_format(self, client):
        response = client.post(
            "/query",
            json={
                "query": "Tell me about AI companies",
                "response_format": "narrative",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] is not None
        assert data["answer"] != ""
        assert data["metadata"]["synthesis_model"] is not None
