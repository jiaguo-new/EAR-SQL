"""Generate BIRD dev predictions from a running vLLM rollout server."""
from __future__ import annotations
import argparse
import json
import os
import sys
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_bird, build_prompt
from src.policy import _extract_sql


def gen_preds(vllm_url: str, model_name: str, data: list, max_gen_len: int = 1024,
              temperature: float = 0.0, n: int = 1, out: str = "preds.jsonl"):
    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f:
        for i, ex in enumerate(data):
            prompt = build_prompt(ex)
            try:
                r = requests.post(vllm_url, json={
                    "model": model_name, "prompt": prompt, "n": n,
                    "max_tokens": max_gen_len, "temperature": temperature,
                    "logprobs": 1,
                }, timeout=600)
                r.raise_for_status()
                text = r.json()["choices"][0]["text"]
                sql = _extract_sql(text)
            except Exception as e:
                print(f"[{i}] generation failed: {e}", file=sys.stderr)
                sql = ""
            f.write(json.dumps({"idx": i, "sql": sql}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"generated {i+1}/{len(data)}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/ear_sql.yaml")
    ap.add_argument("--out", default="preds/preds.jsonl")
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-gen-len", type=int, default=1024)
    args = ap.parse_args()

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    data = load_bird(cfg["data"]["eval_bird"], cfg["data"]["db_root"], cfg["data"]["dialect"])
    model_name = cfg["model"]["base"]
    gen_preds(args.vllm_url, model_name, data,
              max_gen_len=args.max_gen_len or cfg["model"].get("max_gen_len", 1024),
              temperature=args.temperature,
              out=args.out)


if __name__ == "__main__":
    main()
