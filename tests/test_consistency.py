"""Test LLM response consistency with temperature=0."""


def test_query_consistency(client):
    """Run the same query multiple times and verify consistent answers."""
    query = {"query": "Show me all companies founded after 2015"}
    responses = []

    # Run the same query 3 times
    for _ in range(3):
        response = client.post("/query", json=query)
        assert response.status_code == 200
        data = response.json()
        responses.append(data)

    # All responses should have the same sources consulted
    sources = [set(r["sources_consulted"]) for r in responses]
    assert all(s == sources[0] for s in sources), "Sources consulted should be consistent"

    # All responses should have the same number of results
    result_counts = [r["metadata"]["total_results"] for r in responses]
    assert all(c == result_counts[0] for c in result_counts), "Result counts should be consistent"

    # With fake provider at temp=0, answers should be identical
    answers = [r.get("answer") for r in responses]
    assert all(a == answers[0] for a in answers), "Answers should be identical with temp=0"


def test_raw_format_consistency(client):
    """Raw format (no LLM synthesis) should be perfectly consistent."""
    query = {"query": "Show me all companies founded after 2015", "response_format": "raw"}
    responses = []

    for _ in range(3):
        response = client.post("/query", json=query)
        assert response.status_code == 200
        data = response.json()
        responses.append(data)

    # Raw results should be identical
    for i in range(1, len(responses)):
        assert responses[i]["results"] == responses[0]["results"], (
            "Raw format should return identical results"
        )
