"""Merge a PEFT LoRA adapter into its base model and save as a standalone model."""
from __future__ import annotations
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading base model from {args.base} (dtype={args.dtype})...")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging and unloading...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.out}...")
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
