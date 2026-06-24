"""Data loading + prompt construction.

Implements an M-Schema-style serialization (XiYan-SQL): tables/columns with
type, primary-key markers, and a few example values, which empirically helps
the model understand the schema.
"""
from __future__ import annotations
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Example:
    db_id: str
    db_path: str
    question: str
    evidence: str          # BIRD external knowledge ("evidence")
    gold_sql: str
    dialect: str = "sqlite"


def load_bird(json_path: str, db_root: str, dialect: str = "sqlite") -> list[Example]:
    with open(json_path) as f:
        data = json.load(f)
    out: list[Example] = []
    for d in data:
        db_id = d["db_id"]
        out.append(Example(
            db_id=db_id,
            db_path=os.path.join(db_root, db_id, f"{db_id}.sqlite"),
            question=d["question"],
            evidence=d.get("evidence", ""),
            gold_sql=d.get("SQL", d.get("query", "")),
            dialect=dialect,
        ))
    return out


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def m_schema(db_path: str, max_values: int = 1) -> str:
    """Serialize a SQLite schema in an M-Schema-like format.

    (DB -> tables -> columns) with type, PK flag, and example values.
    `max_values=1`: a single example value per column keeps the prompt bounded
    (some BIRD DBs with long text columns blow past 8k tokens at max_values=3).
    Replace the value sampling with your own schema-linking output to feed only
    the *linked* subset for large schemas (Spider 2.0).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    lines: list[str] = ["[DB]"]
    for t in tables:
        lines.append(f"[TABLE] {t}")
        cur.execute(f"PRAGMA table_info('{t}')")
        for _cid, name, ctype, _nn, _def, pk in cur.fetchall():
            tag = " (PK)" if pk else ""
            try:
                cur.execute(f"SELECT DISTINCT \"{name}\" FROM \"{t}\" LIMIT {max_values}")
                vals = ", ".join(str(v[0]) for v in cur.fetchall())
            except Exception:  # noqa: BLE001
                vals = ""
            lines.append(f"  - {name}: {ctype}{tag} | e.g. {vals}")
    conn.close()
    return "\n".join(lines)


PROMPT_TMPL = """You are an expert {dialect} engineer. Read the schema and write ONE correct SQL query.
First reason about which tables/columns are relevant and the query plan, then emit the SQL.

# Schema
{schema}

# External knowledge
{evidence}

# Question
{question}

Put your reasoning first, then the final SQL inside <sql> ... </sql>.
"""


def build_prompt(ex: Example, linked_schema: str | None = None) -> str:
    schema = linked_schema if linked_schema is not None else m_schema(ex.db_path)
    return PROMPT_TMPL.format(dialect=ex.dialect, schema=schema,
                              evidence=ex.evidence or "(none)", question=ex.question)
