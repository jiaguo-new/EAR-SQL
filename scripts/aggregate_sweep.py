"""Aggregate the multi-seed EAR-vs-baseline sweep: mean+/-std EX/VES + difficulty buckets."""
import json
import sys
import statistics
from pathlib import Path


def load(run_dir, tag):
    p = Path(run_dir) / "reports" / f"sw_eval_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def ex_ves(rep):
    s = rep["summary"]
    return s["ex_rate"], s.get("ves", 0.0)


def bucket_ex(rep, idx2diff):
    """EX rate per difficulty bucket."""
    out = {}
    by = {}
    for r in rep["per_query"]:
        d = idx2diff.get(r["idx"], "?")
        by.setdefault(d, []).append(1 if r["ex"] else 0)
    for d, xs in by.items():
        out[d] = 100 * sum(xs) / len(xs)
    return out


def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main():
    run_dir, held = sys.argv[1], sys.argv[2]
    held_data = json.loads(Path(held).read_text())
    idx2diff = {i: d.get("difficulty", "?") for i, d in enumerate(held_data)}

    orig = load(run_dir, "original")
    print("=== Multi-seed sweep: EAR vs baseline (Qwen2.5-Coder-7B, heldout300) ===\n")
    if orig:
        ex, ves = ex_ves(orig)
        print(f"original (untrained): EX {ex:.2f}%  VES {ves:.2f}")
        print(f"  by difficulty: { {k: round(v,1) for k,v in bucket_ex(orig, idx2diff).items()} }\n")

    for name in ("baseline", "ear"):
        exs, vess, buckets = [], [], {}
        for s in (1, 2, 3):
            rep = load(run_dir, f"{name}_s{s}")
            if not rep:
                continue
            ex, ves = ex_ves(rep)
            exs.append(ex); vess.append(ves)
            for d, v in bucket_ex(rep, idx2diff).items():
                buckets.setdefault(d, []).append(v)
        e, v = msd(exs), msd(vess)
        print(f"{name}-trained  (n={len(exs)} seeds): "
              f"EX {e[0]:.2f}±{e[1]:.2f}%  VES {v[0]:.2f}±{v[1]:.2f}" if e else f"{name}: no data")
        if e:
            bstr = {d: f"{statistics.mean(vs):.1f}±{(statistics.stdev(vs) if len(vs)>1 else 0):.1f}"
                    for d, vs in buckets.items()}
            print(f"  EX by difficulty: {bstr}")
            print(f"  per-seed EX: {[round(x,2) for x in exs]}\n")

    # verdict
    be = [ex_ves(load(run_dir, f"baseline_s{s}"))[0] for s in (1,2,3) if load(run_dir, f"baseline_s{s}")]
    ee = [ex_ves(load(run_dir, f"ear_s{s}"))[0] for s in (1,2,3) if load(run_dir, f"ear_s{s}")]
    if be and ee:
        mb, me = statistics.mean(be), statistics.mean(ee)
        print(f"VERDICT: EAR {me:.2f}% vs baseline {mb:.2f}%  (delta {me-mb:+.2f}pp)")
        if len(be) > 1 and len(ee) > 1:
            import math
            sb, se = statistics.stdev(be), statistics.stdev(ee)
            pooled = math.sqrt((sb**2 + se**2) / 2) or 1e-9
            print(f"  std baseline {sb:.2f}, ear {se:.2f}; |delta|/pooled_std = {abs(me-mb)/pooled:.2f}")
            print("  (rule of thumb: needs >~2 to claim a real effect at n=3)")


if __name__ == "__main__":
    main()
