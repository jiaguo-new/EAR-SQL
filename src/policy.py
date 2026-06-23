"""Concrete Policy backend for EAR-SQL.

Generation is served by a vLLM OpenAI-compatible endpoint (fast rollouts with
token logprobs); the PPO-clip update runs on a HuggingFace actor with the
*separated* advantage streams produced by `resampling.compute_advantages`.

This replaces the stub `Policy` in grpo_trainer.py. It is intentionally
framework-light (no TRL/verl lock-in). Swap in verl/OpenRLHF later if you want
their distributed machinery — the interface (generate_group / generate_from_prefix
/ ppo_update) stays the same.

Run a rollout server first, e.g.:
    python -m vllm.entrypoints.openai.api_server \
        --model $BASE_MODEL --enable-prefix-caching --port 8000
"""
from __future__ import annotations
import os
import re
from typing import List

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .resampling import Rollout, AdvantageBundle

SQL_RE = re.compile(r"<sql>(.*?)</sql>", re.S)


def _extract_sql(text: str) -> str:
    m = SQL_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: last ```sql block or trailing line
    return text.strip().splitlines()[-1] if text.strip() else ""


def _is_lora(path: str | None) -> bool:
    if not path:
        return False
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_actor(
    sft_init: str,
    base: str,
    device: str,
    torch_dtype=torch.bfloat16,
):
    """Load the trainable HF actor.

    - If `sft_init` is a PEFT adapter, load `base` and attach the adapter.
      Base weights are frozen; only LoRA parameters are trainable.
    - Otherwise treat `sft_init` as a full merged model and load it directly.
    - If `sft_init` is missing or equals `base`, also treat as a full model.
    """
    if sft_init == base or not _is_lora(sft_init):
        actor = AutoModelForCausalLM.from_pretrained(
            sft_init or base, dtype=torch_dtype
        )
        return actor.to(device)

    import peft

    base_model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch_dtype
    ).to(device)
    actor = peft.PeftModel.from_pretrained(base_model, sft_init)
    # PEFT already marks LoRA params trainable and base params frozen; be explicit
    # in case a merged/standalone adapter was supplied.
    for name, param in actor.named_parameters():
        param.requires_grad = "lora_" in name.lower()
    return actor


class Policy:
    def __init__(self, cfg: dict, load_actor: bool | None = None):
        m = cfg["model"]
        self.cfg = cfg
        self.vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions")
        self.model_name = m["base"]
        self.max_gen = m["max_gen_len"]
        self.temperature = cfg["grpo"]["temperature"]

        # Caches for linking AdvantageBundle.base back to the original rollouts.
        self._last_groups: list[list[Rollout]] = []

        # Actor loading can be skipped for pure inference to save GPU memory.
        if load_actor is None:
            load_actor = os.environ.get("EAR_LOAD_ACTOR", "1").lower() not in (
                "0", "false", "off", "no"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(m.get("sft_init") or m["base"])

        self.actor: torch.nn.Module | None = None
        self.opt: torch.optim.Optimizer | None = None

        if load_actor:
            self.actor = _load_actor(
                m.get("sft_init"), m["base"], self.device
            )
            # Use SGD to keep optimizer-state memory low on a single-card 14B run.
            # (AdamW would need ~2x extra model-size memory for states.)
            trainable = [p for p in self.actor.parameters() if p.requires_grad]
            self.opt = torch.optim.SGD(
                trainable, lr=cfg["grpo"]["lr"], momentum=0.9
            )
            # trade compute for memory during the backward pass
            self.actor.gradient_checkpointing_enable()

        self.clip_eps = cfg["grpo"]["clip_eps"]

    # ---- generation (vLLM) -------------------------------------------------
    def _vllm(self, prompt: str, n: int) -> List[dict]:
        r = requests.post(self.vllm_url, json={
            "model": self.model_name, "prompt": prompt, "n": n,
            "max_tokens": self.max_gen, "temperature": self.temperature,
            "logprobs": 1,
        }, timeout=600)
        r.raise_for_status()
        return r.json()["choices"]

    def _to_rollout(self, prompt: str, choice: dict, prompt_id: int = -1) -> Rollout:
        text = choice["text"]
        # token logprobs over the whole generation; the trainer/split decides the action span
        lp = choice.get("logprobs", {}) or {}
        token_lps = lp.get("token_logprobs", []) or []
        sql = _extract_sql(text)
        # action span = from <sql> onward; approximate via char split then reuse tokens
        idx = text.find("<sql>")
        prefix = text[:idx] if idx != -1 else text
        action = text[idx:] if idx != -1 else ""
        # logprobs for action tokens (approx: tail of the token_logprobs)
        action_lps = token_lps[-max(1, len(self.tok(action).input_ids)):] if token_lps else []
        return Rollout(prompt_id, text, prefix, action, action_lps, sql)

    def generate_group(self, prompt: str, g: int, temperature: float = 1.0) -> List[Rollout]:
        rollouts = [self._to_rollout(prompt, c) for c in self._vllm(prompt, g)]
        self._last_groups.append(rollouts)
        return rollouts

    def generate_from_prefix(self, prompt: str, prefix_text: str, k: int) -> List[Rollout]:
        # condition on prompt + fixed prefix; vLLM prefix caching makes this cheap
        cond = prompt + prefix_text
        outs = []
        for c in self._vllm(cond, k):
            full = prefix_text + c["text"]
            outs.append(self._to_rollout(prompt, {"text": full, "logprobs": c.get("logprobs", {})}))
        return outs

    # ---- PPO-clip update with separated advantage streams ------------------
    def _logprobs_for(self, text: str) -> torch.Tensor:
        if self.actor is None:
            raise RuntimeError("actor is not loaded; cannot compute logprobs")
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        # bfloat16 autocast is GPU-only in older PyTorch; skip it on CPU.
        if self.device == "cpu":
            out = self.actor(ids)
        else:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                out = self.actor(ids)
        logp = F.log_softmax(out.logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]

    def _clip_loss(self, text: str, adv: float, old_lp: list[float] | None = None) -> torch.Tensor:
        new_lp = self._logprobs_for(text)
        if old_lp:
            # vLLM only gives us logprobs for the generated tokens; align them to
            # the suffix of the actor's output (action tokens are at the end).
            old_len = min(len(old_lp), new_lp.shape[0])
            old = torch.tensor(old_lp[-old_len:], device=self.device)
            new_lp = new_lp[-old_len:]
            ratio = torch.exp(new_lp - old)
        else:
            ratio = torch.exp(new_lp - new_lp.detach())  # ratio≈1 at first step
        a = torch.tensor(float(adv), device=self.device)
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * a
        return -torch.min(unclipped, clipped).mean()

    def ppo_update(self, advantages: AdvantageBundle) -> dict:
        if self.actor is None or self.opt is None:
            return {"loss": 0.0, "updated_spans": 0}

        self.opt.zero_grad()
        loss = torch.tensor(0.0, device=self.device)
        n = 0

        # 1) standard GRPO update on non-all-wrong groups using the original rollouts
        for prompt_id, advs in advantages.base.items():
            if prompt_id < 0 or prompt_id >= len(self._last_groups):
                continue
            rollouts = self._last_groups[prompt_id]
            for r, a in zip(rollouts, advs):
                loss = loss + self._clip_loss(r.text, a, r.action_token_logprobs)
                n += 1

        # 2) resampled suffixes: per-prefix advantage on the suffix tokens
        for rr, advs in advantages.suffix:
            for cand, a in zip(rr.resampled, advs):
                loss = loss + self._clip_loss(cand.text, a, cand.action_token_logprobs)
                n += 1

        # 3) source prefixes: recovery-reward advantage on the prefix tokens
        for rr, a in advantages.prefix:
            loss = loss + self._clip_loss(rr.prefix_text, a)
            n += 1

        if n:
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt.step()

        # clear the rollout cache so the next training step starts fresh
        self._last_groups.clear()
        return {"loss": float(loss.detach()) / max(n, 1), "updated_spans": n}
