import json

import pytest

from brains.llm import create_provider
from brains.llm.fake_provider import FakeLLMProvider
from brains.llm.provider import LLMProvider, LLMResponse


class TestFakeLLMProvider:
    def test_fake_provider_implements_protocol(self):
        provider = FakeLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_fake_provider_returns_canned_response(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4o-mini",
        )
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.model == "fake"

    def test_fake_provider_returns_structured_query_plan(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[
                {"role": "system", "content": "You are a query planner."},
                {"role": "user", "content": "What AI companies were founded after 2015?"},
            ],
            model="gpt-4o-mini",
            response_format="json",
        )
        parsed = json.loads(response.content)
        assert "queries" in parsed
        assert isinstance(parsed["queries"], list)

    def test_fake_provider_routes_company_queries_to_sql(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[
                {"role": "system", "content": "You are a query planner."},
                {"role": "user", "content": "What companies were founded after 2015?"},
            ],
            model="gpt-4o-mini",
            response_format="json",
        )
        parsed = json.loads(response.content)
        sources = [q["source"] for q in parsed["queries"]]
        assert "sql" in sources

    def test_fake_provider_routes_conceptual_queries_to_vector(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[
                {"role": "system", "content": "You are a query planner."},
                {"role": "user", "content": "What is machine learning?"},
            ],
            model="gpt-4o-mini",
            response_format="json",
        )
        parsed = json.loads(response.content)
        sources = [q["source"] for q in parsed["queries"]]
        assert "vector" in sources

    def test_fake_provider_routes_relationship_queries_to_graph(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[
                {"role": "system", "content": "You are a query planner."},
                {"role": "user", "content": "What is related to Python?"},
            ],
            model="gpt-4o-mini",
            response_format="json",
        )
        parsed = json.loads(response.content)
        sources = [q["source"] for q in parsed["queries"]]
        assert "graph" in sources

    def test_fake_provider_synthesis_response_is_valid_json(self):
        provider = FakeLLMProvider()
        response = provider.complete(
            messages=[
                {"role": "system", "content": "Synthesize these results."},
                {"role": "user", "content": "Some results context here"},
            ],
            model="gpt-4o-mini",
            response_format="json",
        )
        parsed = json.loads(response.content)
        assert "answer" in parsed
        assert isinstance(parsed["answer"], str)
        assert "confidence" in parsed
        assert 0 <= parsed["confidence"] <= 1


class TestCreateProvider:
    def test_create_fake_provider(self):
        provider = create_provider("fake")
        assert isinstance(provider, FakeLLMProvider)

    def test_create_openai_provider_with_key(self):
        from brains.llm.openai_provider import OpenAIProvider

        provider = create_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)

    def test_create_openai_provider_without_key_raises(self):
        with pytest.raises(ValueError, match="API key is required"):
            create_provider("openai", api_key="")

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider("unknown")
