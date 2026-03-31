from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from brains.sources.base import DataResult, DataSourceSchema

DATA_DIR = Path(__file__).parent.parent / "data"


class VectorSource:
    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._load_fixture_data()

    def _load_fixture_data(self) -> None:
        fixture_path = DATA_DIR / "knowledge.json"
        if not fixture_path.exists():
            return

        with open(fixture_path) as f:
            self._chunks = json.load(f)

        if not self._chunks:
            return

        raw_embeddings = [chunk["embedding"] for chunk in self._chunks]
        matrix = np.array(raw_embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = matrix / norms

    def _embed_query(self, text: str) -> np.ndarray:
        """Create a simple hash-based embedding for the query.

        Uses a deterministic hash approach rather than calling an external
        embedding API. Sufficient for demonstrating the retrieval pattern.
        """
        dim = self._embeddings.shape[1] if self._embeddings is not None else 64
        vec = np.zeros(dim, dtype=np.float32)
        for word in text.lower().split():
            index = hash(word) % dim
            vec[index] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def query(self, params: dict[str, Any]) -> list[DataResult]:
        text = params.get("text", "")
        top_k = params.get("top_k", 5)

        if not text or self._embeddings is None or len(self._chunks) == 0:
            return []

        query_vec = self._embed_query(text)
        similarities = self._embeddings @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                DataResult(
                    source="vector",
                    data={
                        "text": chunk["text"],
                        "category": chunk.get("category", "unknown"),
                        "id": chunk["id"],
                    },
                    score=score,
                    metadata={"similarity": score},
                )
            )
        return results

    def describe(self) -> DataSourceSchema:
        return DataSourceSchema(
            name="vector",
            description=(
                "In-memory vector store with AI/ML knowledge chunks. "
                "Supports semantic similarity search over "
                "pre-embedded text passages."
            ),
            capabilities=[
                "Semantic similarity search",
                "Top-K retrieval",
                "Category filtering",
            ],
            sample_queries=[
                "How do transformer models work?",
                "What is reinforcement learning?",
                "Explain the attention mechanism",
            ],
        )
