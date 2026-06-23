"""Reward functions.

Primary signal is execution correctness (Arctic-style minimal reward). We also
expose the *recovery reward* used to credit a fixed prefix when any of its
resampled SQL actions succeeds (AXPO core).
"""
from __future__ import annotations
from typing import Sequence
import sqlglot

from .sql_exec import execute, rows_match, ExecResult


def syntax_valid(sql: str, dialect: str = "sqlite") -> bool:
    try:
        sqlglot.parse_one(sql, read=dialect)
        return True
    except Exception:  # noqa: BLE001
        return False


def execution_reward(pred_sql: str, gold_rows, db_path: str, dialect: str = "sqlite",
                     exec_weight: float = 1.0, syntax_weight: float = 0.0,
                     timeout_s: int = 30) -> float:
    """0/1 execution reward, with optional small syntax partial credit.

    Keep `syntax_weight=0` for the pure Arctic-style minimal reward; raise it
    only as an auxiliary dense term in ablations.
    """
    res: ExecResult = execute(db_path, pred_sql, dialect, timeout_s)
    if res.ok and rows_match(res.rows, gold_rows):
        return exec_weight
    if syntax_weight and syntax_valid(pred_sql, dialect):
        return syntax_weight
    return 0.0


def recovery_reward(resampled_rewards: Sequence[float]) -> float:
    """AXPO recovery indicator: prefix is 'good' if >=1 resample succeeded.

    r_prefix = 1[ exists k : reward_k > 0 ]
    Replaces the original (zero) reward of the all-wrong source group so the
    prefix tokens receive a positive learning signal.
    """
    return 1.0 if any(r > 0 for r in resampled_rewards) else 0.0
