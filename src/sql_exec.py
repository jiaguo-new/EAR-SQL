"""SQL execution + result-set comparison for execution-based reward.

Kept dialect-agnostic via a thin `execute()` entry point. SQLite is implemented;
BigQuery/Snowflake (Spider 2.0) are stubbed.
"""
from __future__ import annotations
import sqlite3
from typing import Any, Sequence
from func_timeout import func_timeout, FunctionTimedOut


class ExecResult:
    def __init__(self, ok: bool, rows: Sequence[tuple] | None = None, error: str | None = None):
        self.ok = ok
        self.rows = rows
        self.error = error


def _run_sqlite(db_path: str, sql: str) -> ExecResult:
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return ExecResult(True, rows=rows)
    except Exception as e:  # noqa: BLE001
        return ExecResult(False, error=str(e))


def execute(db_path: str, sql: str, dialect: str = "sqlite", timeout_s: int = 30) -> ExecResult:
    """Execute `sql` with a hard timeout; never raise to the trainer."""
    runner = {
        "sqlite": _run_sqlite,
        # "bigquery": _run_bigquery,   # TODO Spider 2.0
        # "snowflake": _run_snowflake, # TODO Spider 2.0
    }.get(dialect)
    if runner is None:
        return ExecResult(False, error=f"dialect {dialect} not implemented")
    try:
        return func_timeout(timeout_s, runner, args=(db_path, sql))
    except FunctionTimedOut:
        return ExecResult(False, error="timeout")


def rows_match(pred: Sequence[tuple] | None, gold: Sequence[tuple] | None,
               order_sensitive: bool = False) -> bool:
    """Result-set equality used by BIRD/Spider execution accuracy.

    BIRD compares as a set unless the query has ORDER BY. Keep this consistent
    with the official evaluator you adopt.
    """
    if pred is None or gold is None:
        return False
    if order_sensitive:
        return list(pred) == list(gold)
    return _as_multiset(pred) == _as_multiset(gold)


def _as_multiset(rows: Sequence[tuple]) -> dict[tuple, int]:
    out: dict[tuple, int] = {}
    for r in rows:
        key = tuple(r)
        out[key] = out.get(key, 0) + 1
    return out
