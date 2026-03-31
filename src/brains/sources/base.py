from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class DataSourceSchema:
    name: str
    description: str
    capabilities: list[str]
    sample_queries: list[str]


@dataclass
class DataResult:
    source: str
    data: dict[str, Any]
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataSource(Protocol):
    def query(self, params: dict[str, Any]) -> list[DataResult]: ...
    def describe(self) -> DataSourceSchema: ...
