from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from brains.sources.base import DataResult, DataSourceSchema

DATA_DIR = Path(__file__).parent.parent / "data"


class GraphSource:
    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._node_attrs: dict[str, dict] = {}
        self._load_fixture_data()

    def _load_fixture_data(self) -> None:
        fixture_path = DATA_DIR / "graph.json"
        if not fixture_path.exists():
            return

        with open(fixture_path) as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            node_id = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            self._graph.add_node(node_id, **attrs)
            self._node_attrs[node_id] = attrs

        for edge in data.get("edges", []):
            self._graph.add_edge(
                edge["from"],
                edge["to"],
                relation=edge.get("relation", "related_to"),
            )

        self._node_lookup = {n.lower(): n for n in self._graph.nodes()}

    def _resolve_node(self, name: str) -> str:
        """Case-insensitive node lookup with exact match fallback."""
        return self._node_lookup.get(name.strip().lower(), name)

    def query(self, params: dict[str, Any]) -> list[DataResult]:
        operation = params.get("operation", "neighbors")

        if operation == "neighbors":
            return self._query_neighbors(params)
        elif operation == "path":
            return self._query_path(params)
        elif operation == "subgraph":
            return self._query_subgraph(params)
        return []

    def _query_neighbors(self, params: dict[str, Any]) -> list[DataResult]:
        node = self._resolve_node(params.get("node", ""))
        if node not in self._graph:
            return []

        results = []
        for _, neighbor, edge_data in self._graph.out_edges(node, data=True):
            results.append(
                DataResult(
                    source="graph",
                    data={
                        "neighbor": neighbor,
                        "relation": edge_data.get("relation", "related_to"),
                        "direction": "outgoing",
                        **self._node_attrs.get(neighbor, {}),
                    },
                    score=1.0,
                )
            )
        for predecessor, _, edge_data in self._graph.in_edges(node, data=True):
            results.append(
                DataResult(
                    source="graph",
                    data={
                        "neighbor": predecessor,
                        "relation": edge_data.get("relation", "related_to"),
                        "direction": "incoming",
                        **self._node_attrs.get(predecessor, {}),
                    },
                    score=1.0,
                )
            )
        return results

    def _query_path(self, params: dict[str, Any]) -> list[DataResult]:
        source_node = self._resolve_node(params.get("from", ""))
        target_node = self._resolve_node(params.get("to", ""))

        if source_node not in self._graph or target_node not in self._graph:
            return []

        try:
            path = nx.shortest_path(self._graph.to_undirected(), source_node, target_node)
        except nx.NetworkXNoPath:
            return []

        return [
            DataResult(
                source="graph",
                data={
                    "path": path,
                    "length": len(path) - 1,
                    "from": source_node,
                    "to": target_node,
                },
                score=1.0 / len(path),
            )
        ]

    def _query_subgraph(self, params: dict[str, Any]) -> list[DataResult]:
        node = self._resolve_node(params.get("node", ""))
        depth = params.get("depth", 1)

        if node not in self._graph:
            return []

        visited = {node}
        frontier = [node]
        for _ in range(depth):
            next_frontier = []
            for n in frontier:
                for neighbor in set(self._graph.successors(n)) | set(
                    self._graph.predecessors(n)
                ):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        return [
            DataResult(
                source="graph",
                data={
                    "node": n,
                    **self._node_attrs.get(n, {}),
                },
                score=1.0,
            )
            for n in visited
        ]

    def describe(self) -> DataSourceSchema:
        nodes = sorted(self._graph.nodes())
        return DataSourceSchema(
            name="graph",
            description=(
                f"Technology relationship graph with "
                f"{self._graph.number_of_nodes()} nodes and "
                f"{self._graph.number_of_edges()} edges. "
                f"Nodes: {', '.join(nodes)}. "
                f"Relations: implemented_in, based_on, created, uses, stored_in, applied_to. "
                f"Operations: neighbors (connected nodes), path (shortest path), subgraph (BFS). "
                f'Example: {{"operation": "neighbors", "node": "PyTorch"}}'
            ),
            capabilities=["Neighbor lookup", "Shortest path", "Subgraph extraction"],
            sample_queries=[
                '{"operation": "neighbors", "node": "Python"}',
                '{"operation": "path", "from": "PyTorch", "to": "Google"}',
                '{"operation": "subgraph", "node": "Transformers", "depth": 2}',
            ],
        )
