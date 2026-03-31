import os

import pytest
from fastapi.testclient import TestClient

collect_ignore = []
if os.environ.get("BRAINS_SMOKE_TEST") != "1":
    collect_ignore.append("test_smoke.py")


@pytest.fixture
def client(monkeypatch):
    """Test client with fake LLM provider."""
    monkeypatch.setenv("BRAINS_LLM_PROVIDER", "fake")
    monkeypatch.setenv("BRAINS_OPENAI_API_KEY", "not-needed")

    from brains.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
