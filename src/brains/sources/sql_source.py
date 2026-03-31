from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from brains.sources.base import DataResult, DataSourceSchema

DATA_DIR = Path(__file__).parent.parent / "data"
ALLOWED_STATEMENTS = {"SELECT"}


class SQLiteSource:
    def __init__(self, db_path: Path | None = None):
        self._db_path = str(db_path) if db_path else ":memory:"
        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row
        self._load_fixture_data()

    def _load_fixture_data(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                name TEXT PRIMARY KEY,
                sector TEXT NOT NULL,
                founded INTEGER NOT NULL,
                headquarters TEXT NOT NULL,
                employees INTEGER NOT NULL,
                description TEXT NOT NULL
            )
        """)
        fixture_path = DATA_DIR / "companies.json"
        if fixture_path.exists():
            with open(fixture_path) as f:
                companies = json.load(f)
            for company in companies:
                cursor.execute(
                    "INSERT OR IGNORE INTO companies VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        company["name"],
                        company["sector"],
                        company["founded"],
                        company["headquarters"],
                        company["employees"],
                        company["description"],
                    ),
                )
        self._connection.commit()

    def query(self, params: dict[str, Any]) -> list[DataResult]:
        sql = params.get("sql", "")
        statement_type = sql.strip().split()[0].upper() if sql.strip() else ""
        if statement_type not in ALLOWED_STATEMENTS:
            return []

        try:
            cursor = self._connection.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
        except sqlite3.Error:
            return []

        return [
            DataResult(
                source="sql",
                data=dict(row),
                score=1.0,
                metadata={"query": sql},
            )
            for row in rows
        ]

    def describe(self) -> DataSourceSchema:
        return DataSourceSchema(
            name="sql",
            description=(
                "SQLite relational database with tech companies data. "
                "Table: companies (name, sector, founded, headquarters, "
                "employees, description)."
            ),
            capabilities=["SQL SELECT queries", "Filtering", "Aggregation", "Sorting"],
            sample_queries=[
                "SELECT * FROM companies WHERE sector = 'AI'",
                "SELECT name, founded FROM companies WHERE founded > 2015 ORDER BY founded",
                "SELECT sector, COUNT(*) as count FROM companies GROUP BY sector",
            ],
        )
