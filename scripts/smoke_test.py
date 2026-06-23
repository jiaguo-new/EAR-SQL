"""EAR-SQL smoke test — verifies the implemented components on a real SQLite DB.

Runs with no GPU and no model. Proves: SQL execution + result-set match, the
execution reward, the recovery reward, and the full action-resampling pipeline
(all-wrong detection -> prefix selection -> resample -> recovery -> advantages).

Exit code 0 = all green.
"""
from __future__ import annotations
import os, sys, tempfile, sqlite3, types

# make optional deps soft so the smoke test runs even before pip completes
for _m, _attrs in {
    "func_timeout": {"func_timeout": lambda t, f, args=(): f(*args), "FunctionTimedOut": type("FT", (Exception,), {})},
    "sqlglot": {"parse_one": lambda *a, **k: True},
}.items():
    try:
        __import__(_m)
    except Exception:
        mod = types.ModuleType(_m)
        for k, v in _attrs.items():
            setattr(mod, k, v)
        sys.modules[_m] = mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ear_sql/
from src.sql_exec import execute, rows_match              # noqa: E402
from src.reward import execution_reward, recovery_reward  # noqa: E402
from src.resampling import (                              # noqa: E402
    Rollout, Group, find_allwrong_groups, split_prefix_action,
    select_prefixes, resample_actions, compute_advantages,
)

PASS, FAIL = "\033[1;32mPASS\033[0m", "\033[1;31mFAIL\033[0m"
n_fail = 0
def check(name, cond):
    global n_fail
    print(f"  [{PASS if cond else FAIL}] {name}")
    if not cond: n_fail += 1

def make_db():
    path = os.path.join(tempfile.mkdtemp(), "shop.sqlite")
    c = sqlite3.connect(path); cur = c.cursor()
    cur.executescript("""
        CREATE TABLE customer(id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, cid INTEGER, amount REAL);
        INSERT INTO customer VALUES (1,'Ann','NYC'),(2,'Bob','LA'),(3,'Cy','NYC');
        INSERT INTO orders VALUES (1,1,100),(2,1,50),(3,2,200),(4,3,30);
    """)
    c.commit(); c.close(); return path

def main():
    print("== 1. SQL execution + result-set match ==")
    db = make_db()
    gold = "SELECT name FROM customer WHERE city='NYC' ORDER BY name"
    g = execute(db, gold); check("gold executes", g.ok)
    good = execute(db, "SELECT name FROM customer WHERE city='NYC' ORDER BY name")
    check("correct SQL matches gold", rows_match(good.rows, g.rows))
    bad = execute(db, "SELECT name FROM customer WHERE city='LA'")
    check("wrong SQL does NOT match", not rows_match(bad.rows, g.rows))
    broken = execute(db, "SELECT nope FROM nope")
    check("broken SQL returns not-ok", not broken.ok)

    print("== 2. rewards ==")
    gold_rows = g.rows
    check("exec reward = 1 for correct",
          execution_reward(gold, gold_rows, db) == 1.0)
    check("exec reward = 0 for wrong",
          execution_reward("SELECT name FROM customer WHERE city='LA'", gold_rows, db) == 0.0)
    check("recovery reward fires if any resample ok", recovery_reward([0, 0, 1, 0]) == 1.0)
    check("recovery reward 0 if all fail", recovery_reward([0, 0, 0]) == 0.0)

    print("== 3. action-resampling pipeline (all-wrong recovery) ==")
    def mk(pid, lp, rew, sql):
        r = Rollout(pid, f"plan...<sql>{sql}", "plan...", f"<sql>{sql}", [lp], sql)
        r.reward = rew; return r
    # an all-wrong group (3 failing rollouts, varying confidence) + a mixed group
    g0 = Group(0, [mk(0,-0.1,0,"SELECT 0"), mk(0,-2.0,0,"SELECT 0"), mk(0,-0.5,0,"SELECT 0")])
    g1 = Group(1, [mk(1,-0.1,1,"SELECT name FROM customer"), mk(1,-0.3,0,"x")])
    aw = find_allwrong_groups([g0, g1])
    check("detects exactly the all-wrong group", len(aw) == 1 and aw[0].prompt_id == 0)

    pfx, act = split_prefix_action("reason here <sql>SELECT 1</sql>")
    check("prefix/action split at <sql>", pfx == "reason here " and act.startswith("<sql>"))

    chosen = select_prefixes(aw, base_budget=10, budget_ratio=0.25, K=2)
    check("budget picks the most-uncertain prefix", bool(chosen) and chosen[0].action_token_logprobs == [-2.0])

    # resample: pretend the policy now finds a correct SQL on attempt 2
    def fake_gen(prefix_text, k):
        outs = []
        for i in range(k):
            sql = gold if i == 1 else "SELECT 0"
            outs.append(Rollout(0, prefix_text+sql, prefix_text, sql, [-0.2], sql))
        return outs
    rr = resample_actions(chosen[0], K=2,
                          generate_from_prefix=fake_gen,
                          score_reward=lambda c: execution_reward(c.sql, gold_rows, db))
    check("resampling recovers a correct trajectory", rr.recovery == 1.0)

    adv = compute_advantages([g0, g1], [rr])
    check("all-wrong group excluded from base advantages", 0 not in adv.base and 1 in adv.base)
    check("one suffix + one prefix advantage stream produced",
          len(adv.suffix) == 1 and len(adv.prefix) == 1)

    print()
    if n_fail == 0:
        print("\033[1;32mSMOKE TEST: ALL GREEN — exec/reward/resampling verified on this machine.\033[0m")
        return 0
    print(f"\033[1;31mSMOKE TEST: {n_fail} check(s) failed.\033[0m")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
