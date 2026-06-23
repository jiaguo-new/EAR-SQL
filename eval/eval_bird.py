"""Robust BIRD evaluation: EM / EX / Valid SQL Rate / JOIN EX.

Each SQL runs in a dedicated child process with a hard wall-clock timeout,
so a runaway query can never block the evaluator via the SQLite GIL.

Usage:
    python eval/eval_bird.py \
        --pred preds.jsonl \
        --gold data/bird/dev.json \
        --db_root data/bird/dev_databases \
        --output reports/eval_bird.json \
        --timeout 15 \
        --limit 5000

``--pred`` accepts either:
  - a JSONL file where each line is {"idx": int, "sql": str}, or
  - a plain text file with one SQL per line (lines map to gold by index).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import multiprocessing.pool
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from src.sql_exec import execute, rows_match, ExecResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PRINT_EVERY = 50
QUEUE_GET_TIMEOUT = 5  # seconds to wait for the worker result queue
KILL_WAIT = 2          # seconds to wait after terminate before kill
MAX_WORKERS = max(1, min(mp.cpu_count(), 8))


class _NonDaemonProcess(mp.get_context("spawn").Process):
    """A non-daemonic Process that ignores attempts to mark it daemon."""

    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value: object) -> None:
        pass


class _NonDaemonPool(mp.pool.Pool):
    """Pool whose workers are not daemonic, so they can spawn their own children."""

    @staticmethod
    def Process(ctx: mp.context.BaseContext, *args: object, **kwargs: object):
        return _NonDaemonProcess(*args, **kwargs)


# ---------------------------------------------------------------------------
# Result-set normalization / comparison
# ---------------------------------------------------------------------------
def _normalize_cell(v: Any) -> Any:
    """Normalize a single result cell for multiset comparison."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 3)
    return str(v).strip().lower()


def _normalize_rows(rows: Sequence[tuple[Any, ...]] | None) -> list[tuple[Any, ...]]:
    """Normalize every cell in a result set."""
    if rows is None:
        return []
    return [tuple(_normalize_cell(v) for v in row) for row in rows]


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for Exact Match comparison."""
    return " ".join(sql.strip().lower().rstrip(";").split())


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------
def _worker_main(args: tuple[int, dict[str, Any], str, str, int, int],
                 queue: mp.queues.Queue) -> None:
    """Run in child process: execute gold + pred and compare."""
    idx, item, pred_sql, db_root, limit, timeout = args
    db_id = item.get("db_id", "")
    db_path = str(Path(db_root) / db_id / f"{db_id}.sqlite")
    gold_sql = item.get("SQL") or item.get("query", "") or ""
    is_join = bool(re.search(r"\bjoin\b", gold_sql, re.I))

    per_sql_timeout = max(1, timeout // 2)

    try:
        gold_res = execute(db_path, gold_sql, timeout_s=per_sql_timeout)
        if not pred_sql or not pred_sql.strip():
            pred_res = ExecResult(False, error="empty_sql")
        else:
            pred_res = execute(db_path, pred_sql, timeout_s=per_sql_timeout)

        em = _normalize_sql(pred_sql) == _normalize_sql(gold_sql)
        ex = False
        if gold_res.ok and pred_res.ok:
            gold_rows = gold_res.rows[:limit] if gold_res.rows else None
            pred_rows = pred_res.rows[:limit] if pred_res.rows else None
            ex = rows_match(
                _normalize_rows(pred_rows),
                _normalize_rows(gold_rows),
            )

        result: dict[str, Any] = {
            "idx": idx,
            "db_id": db_id,
            "ex": ex,
            "em": em,
            "valid": pred_res.ok,
            "gold_error": gold_res.error if not gold_res.ok else None,
            "pred_error": pred_res.error if not pred_res.ok else None,
            "is_join": is_join,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "idx": idx,
            "db_id": db_id,
            "ex": False,
            "em": False,
            "valid": False,
            "gold_error": None,
            "pred_error": f"worker_crash: {exc}",
            "is_join": is_join,
            "worker_error": str(exc),
        }

    queue.put(result)


def _evaluate_one(task: tuple[int, dict[str, Any], str, str, int, int]) -> dict[str, Any]:
    """Spawn a single child process and enforce a hard timeout."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker_main, args=(task, queue))
    proc.start()

    # The task tuple contains the configured per-query timeout.
    timeout = task[-1]
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(KILL_WAIT)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "idx": task[0],
            "db_id": task[1].get("db_id", ""),
            "ex": False,
            "em": False,
            "valid": False,
            "gold_error": None,
            "pred_error": "eval_timeout",
            "is_join": bool(
                re.search(
                    r"\bjoin\b",
                    task[1].get("SQL") or task[1].get("query", ""),
                    re.I,
                )
            ),
        }

    try:
        return queue.get(block=True, timeout=QUEUE_GET_TIMEOUT)
    except mp.queues.Empty:
        logger.warning("Worker for idx %s finished without returning a result", task[0])
        return {
            "idx": task[0],
            "db_id": task[1].get("db_id", ""),
            "ex": False,
            "em": False,
            "valid": False,
            "gold_error": None,
            "pred_error": "eval_no_result",
            "is_join": bool(
                re.search(
                    r"\bjoin\b",
                    task[1].get("SQL") or task[1].get("query", ""),
                    re.I,
                )
            ),
        }
    except Exception:
        logger.exception("Unexpected error reading result queue for idx %s", task[0])
        raise


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load_predictions(pred_path: str) -> dict[int, str]:
    """Load predictions as {idx: sql} from JSONL or plain text."""
    preds: dict[int, str] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    idx = obj.get("idx", line_no)
                    sql = obj.get("sql", "")
                else:
                    idx = line_no
                    sql = str(obj)
            except json.JSONDecodeError:
                idx = line_no
                sql = line
            preds[idx] = sql
    return preds


def _load_gold(gold_path: str) -> list[dict[str, Any]]:
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("data", [])
    return list(data)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    pred_path: str,
    gold_path: str,
    db_root: str,
    output: str,
    timeout: int,
    limit: int,
) -> dict[str, Any]:
    preds = _load_predictions(pred_path)
    gold = _load_gold(gold_path)

    gold_indices = set(range(len(gold)))
    pred_indices = set(preds.keys())
    common_indices = sorted(gold_indices & pred_indices)

    missing_in_pred = sorted(gold_indices - pred_indices)
    extra_in_pred = sorted(pred_indices - gold_indices)

    if missing_in_pred:
        logger.warning(
            "%d gold example(s) missing from predictions (idx: %s)",
            len(missing_in_pred),
            _compact_indices(missing_in_pred),
        )
    if extra_in_pred:
        logger.warning(
            "%d prediction(s) have no matching gold example (idx: %s); "
            "they will NOT be silently dropped.",
            len(extra_in_pred),
            _compact_indices(extra_in_pred),
        )

    if not common_indices:
        raise ValueError("No overlapping examples found between pred and gold.")

    tasks = [
        (i, gold[i], preds[i], db_root, limit, timeout) for i in common_indices
    ]

    ctx = mp.get_context("spawn")
    n = len(tasks)

    print(f"Evaluating {n} BIRD queries with subprocess timeout={timeout}s ...")
    sys.stdout.flush()

    results: list[dict[str, Any]] = []
    with _NonDaemonPool(processes=MAX_WORKERS, context=ctx) as pool:
        async_results = [
            pool.apply_async(_evaluate_one, (task,)) for task in tasks
        ]
        for i, async_res in enumerate(async_results):
            # _evaluate_one already enforces the hard per-query timeout.
            res = async_res.get(timeout=timeout + QUEUE_GET_TIMEOUT + 5)
            if res.get("worker_error"):
                logger.warning(
                    "Worker error for idx %s: %s", res["idx"], res["worker_error"]
                )
            res.pop("worker_error", None)
            results.append(res)
            completed = i + 1
            if completed % PRINT_EVERY == 0 or completed == n:
                interim_ex = sum(1 for r in results if r["ex"])
                interim_valid = sum(1 for r in results if r["valid"])
                print(
                    f"  [{completed}/{n}] EX={interim_ex}/{completed} "
                    f"({100 * interim_ex / completed:.1f}%), "
                    f"Valid={interim_valid}/{completed} "
                    f"({100 * interim_valid / completed:.1f}%)"
                )
                sys.stdout.flush()

    em_count = sum(1 for r in results if r["em"])
    ex_count = sum(1 for r in results if r["ex"])
    valid_count = sum(1 for r in results if r["valid"])
    join_results = [r for r in results if r["is_join"]]
    join_ex = sum(1 for r in join_results if r["ex"])

    join_ex_rate = 100 * join_ex / len(join_results) if join_results else 0.0

    summary = {
        "total": n,
        "em": em_count,
        "ex": ex_count,
        "valid": valid_count,
        "em_rate": 100 * em_count / n,
        "ex_rate": 100 * ex_count / n,
        "valid_rate": 100 * valid_count / n,
        "join_total": len(join_results),
        "join_ex": join_ex,
        "join_ex_rate": join_ex_rate,
    }

    print("\n========== BIRD Evaluation Results ==========")
    print(f"Total queries: {n}")
    print(f"Exact Match (EM): {em_count} / {n} = {summary['em_rate']:.2f}%")
    print(f"Execution Match (EX): {ex_count} / {n} = {summary['ex_rate']:.2f}%")
    print(f"Valid SQL Rate: {valid_count} / {n} = {summary['valid_rate']:.2f}%")
    if join_results:
        print(f"JOIN EX: {join_ex} / {len(join_results)} = {join_ex_rate:.2f}%")
    else:
        print("JOIN EX: N/A (no JOIN queries)")
    print("=============================================")

    report = {"summary": summary, "per_query": results}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nPer-query results written to {output}")

    return report


def _compact_indices(indices: list[int]) -> str:
    """Pretty-print a list of indices for warning messages."""
    if len(indices) <= 10:
        return str(indices)
    return str(indices[:10]) + f", ... ({len(indices) - 10} more)"


def main():
    parser = argparse.ArgumentParser(description="BIRD evaluation (EM/EX/Valid/JOIN EX)")
    parser.add_argument("--pred", required=True, help="Prediction JSONL or plain SQL file.")
    parser.add_argument("--gold", required=True, help="BIRD dev.json.")
    parser.add_argument("--db_root", required=True, help="Root directory containing BIRD databases.")
    parser.add_argument("--output", default="reports/eval_bird.json", help="Path to write JSON report.")
    parser.add_argument("--timeout", type=int, default=15, help="Seconds per SQL execution.")
    parser.add_argument("--limit", type=int, default=5000, help="Max rows to fetch per query.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluate(args.pred, args.gold, args.db_root, args.output, args.timeout, args.limit)


if __name__ == "__main__":
    main()
