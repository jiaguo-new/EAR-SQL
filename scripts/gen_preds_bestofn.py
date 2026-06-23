"""Generate BIRD predictions via best-of-N rollouts, with optional EAR resampling.

This is an *inference-time* reproduction of the baseline vs. EAR-SQL contrast:
  - baseline: pick the best rollout out of G samples.
  - ear:      for all-wrong groups, resample K suffixes per uncertain prefix and
              pick the best among base + resampled rollouts.

Uses direct HTTP calls to the vLLM server to avoid loading the HF actor
(which would compete for GPU memory with the running vLLM engine).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import requests

from src.data import load_bird, build_prompt, Example
from src.reward import execution_reward
from src.resampling import (
    Group, Rollout, find_allwrong_groups, split_prefix_action,
    select_prefixes, resample_actions,
)
from src.sql_exec import execute
from transformers import AutoTokenizer

SQL_RE = re.compile(r"<sql>(.*?)</sql>", re.S)


def extract_sql(text: str) -> str:
    m = SQL_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def vllm_generate(url: str, model: str, prompt: str, n: int, max_tokens: int,
                  temperature: float, seed: int = 42,
                  timeout: float = 600.0) -> list[dict]:
    r = requests.post(url, json={
        "model": model, "prompt": prompt, "n": n,
        "max_tokens": max_tokens, "temperature": temperature,
        "logprobs": 1, "seed": seed,
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"]


def make_rollout(text: str) -> Rollout:
    idx = text.find("<sql>")
    prefix = text[:idx] if idx != -1 else text
    action = text[idx:] if idx != -1 else ""
    return Rollout(0, text, prefix, action, [], extract_sql(text))


def score_rollout(r: Rollout, ex: Example) -> float:
    res = execute(ex.db_path, ex.gold_sql, ex.dialect)
    gold_rows = res.rows if res.ok else None
    return execution_reward(
        r.sql, gold_rows, ex.db_path, ex.dialect,
        exec_weight=1.0, syntax_weight=0.0, timeout_s=30,
    )


def generate_group(url: str, model: str, ex: Example, g: int, temperature: float,
                   max_gen_len: int,
                   build_prompt_fn: typing.Callable[[Example], str] = build_prompt) -> list[Rollout]:
    prompt = build_prompt_fn(ex)
    # vLLM rejects n>1 with temperature=0; fall back to a single greedy rollout.
    n = g if temperature > 0 else 1
    choices = vllm_generate(url, model, prompt, n, max_gen_len, temperature)
    rollouts = [make_rollout(c["text"]) for c in choices]
    for r in rollouts:
        r.reward = score_rollout(r, ex)
    return rollouts


def generate_from_prefix(url: str, model: str, ex: Example, prefix_text: str, k: int,
                         temperature: float, max_gen_len: int,
                         build_prompt_fn: typing.Callable[[Example], str] = build_prompt) -> list[Rollout]:
    prompt = build_prompt_fn(ex) + prefix_text
    n = k if temperature > 0 else 1
    choices = vllm_generate(url, model, prompt, n, max_gen_len, temperature)
    outs = []
    for c in choices:
        full = prefix_text + c["text"]
        outs.append(make_rollout(full))
    for r in outs:
        r.reward = score_rollout(r, ex)
    return outs


def best_sql(rollouts: list[Rollout]) -> str:
    if not rollouts:
        return ""
    best = max(rollouts, key=lambda r: r.reward)
    return best.sql


def process_one(url: str, model: str, max_gen_len: int, ex: Example, cfg: dict,
                ear: bool,
                build_prompt_fn: typing.Callable[[Example], str] = build_prompt) -> tuple[str, str | None, dict]:
    g = cfg["grpo"]["group_size"]
    temp = cfg["grpo"]["temperature"]
    rc = cfg["resample"]

    rollouts = generate_group(url, model, ex, g, temp, max_gen_len, build_prompt_fn=build_prompt_fn)
    base_sql = best_sql(rollouts)
    local = {"all_wrong": 0, "resampled": 0, "recovered": 0}
    if ear and rc.get("enabled", False):
        group = Group(0, rollouts)
        allwrong = find_allwrong_groups([group])
        local["all_wrong"] = len(allwrong)
        chosen = select_prefixes(allwrong, base_budget=g,
                                 budget_ratio=rc["budget_ratio"], K=rc["K"])
        local["resampled"] = len(chosen)
        for prefix_rollout in chosen:
            rr = resample_actions(
                prefix_rollout, rc["K"],
                generate_from_prefix=lambda pfx, k, ex=ex: generate_from_prefix(
                    url, model, ex, pfx, k, temp, max_gen_len,
                    build_prompt_fn=build_prompt_fn),
                score_reward=lambda c: score_rollout(c, ex),
            )
            rollouts.extend(rr.resampled)
            if rr.recovery > 0:
                local["recovered"] += 1
    ear_sql = best_sql(rollouts) if ear else None
    return base_sql, ear_sql, local


def run_stage(url: str, model: str, max_gen_len: int, data: list[Example], cfg: dict,
              ear: bool = False, max_examples: int | None = None,
              workers: int = 4,
              build_prompt_fn: typing.Callable[[Example], str] = build_prompt) -> tuple[list[str], list[str] | None, dict]:
    subset = data[:max_examples] if max_examples else data
    metrics = {"all_wrong_groups": 0, "resampled_prefixes": 0, "recovered": 0, "total": len(subset)}
    preds = [""] * len(subset)
    base_preds = [""] * len(subset) if ear else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, url, model, max_gen_len, e, cfg, ear,
                             build_prompt_fn): i
                   for i, e in enumerate(subset)}
        for fut in as_completed(futures):
            i = futures[fut]
            base_sql, ear_sql, local = fut.result()
            preds[i] = ear_sql if ear else base_sql
            if ear:
                base_preds[i] = base_sql
            metrics["all_wrong_groups"] += local["all_wrong"]
            metrics["resampled_prefixes"] += local["resampled"]
            metrics["recovered"] += local["recovered"]
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(subset)}] aw={metrics['all_wrong_groups']} rec={metrics['recovered']}")
    return preds, base_preds, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/home/dameng/ear_sql/configs/ear_sql.yaml")
    ap.add_argument("--out", default="preds/preds.jsonl")
    ap.add_argument("--ear", action="store_true", help="enable action resampling")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=None,
                    help="override sampling temperature (0.0 = greedy, deterministic)")
    ap.add_argument("--K", type=int, default=None, help="override resample.K")
    ap.add_argument("--budget-ratio", type=float, default=None, help="override resample.budget_ratio")
    ap.add_argument("--metrics", default=None, help="optional JSON file to dump stage metrics")
    ap.add_argument("--baseline-out", default=None, help="when --ear is set, also write baseline best-of-N predictions to this file")
    ap.add_argument("--baseline-metrics", default=None, help="when --ear is set, also write baseline metrics to this file")
    ap.add_argument("--apply-chat-template", action="store_true",
                    help="wrap the prompt with the model's instruction/chat template (requires transformers)")
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    if args.K is not None:
        cfg["resample"]["K"] = args.K
    if args.budget_ratio is not None:
        cfg["resample"]["budget_ratio"] = args.budget_ratio
    if args.temperature is not None:
        cfg["grpo"]["temperature"] = args.temperature

    url = os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions")
    model = cfg["model"]["base"]
    max_gen_len = cfg["model"].get("max_gen_len", 1024)

    build_prompt_fn = build_prompt
    if args.apply_chat_template:
        print(f"Loading tokenizer from {model} ...")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        def _chat_prompt(ex: Example) -> str:
            raw = build_prompt(ex)
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        build_prompt_fn = _chat_prompt

    data = load_bird(cfg["data"]["eval_bird"], cfg["data"]["db_root"], cfg["data"]["dialect"])
    preds, base_preds, metrics = run_stage(url, model, max_gen_len, data, cfg, ear=args.ear,
                                           max_examples=args.max_examples, workers=args.workers,
                                           build_prompt_fn=build_prompt_fn)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for i, sql in enumerate(preds):
            f.write(json.dumps({"idx": i, "sql": sql}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print("metrics:", metrics)

    if args.metrics:
        with open(args.metrics, "w") as f:
            json.dump(metrics, f, indent=2)

    if args.ear and base_preds is not None and args.baseline_out:
        os.makedirs(os.path.dirname(args.baseline_out) or ".", exist_ok=True)
        with open(args.baseline_out, "w") as f:
            for i, sql in enumerate(base_preds):
                f.write(json.dumps({"idx": i, "sql": sql}, ensure_ascii=False) + "\n")
        print(f"wrote baseline {args.baseline_out}")


if __name__ == "__main__":
    main()
