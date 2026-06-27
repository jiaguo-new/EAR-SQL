"""GRPO training loop with the EAR-SQL action-resampling hook.

This is a *skeleton*: rollout generation, logprob extraction, and the optimizer
step are marked `# TODO` for you to bind to your RL stack (TRL / verl / vLLM).
The orchestration — grouping, all-wrong detection, budgeted resampling, and the
separated PPO-clip objective — is written out so the contribution is concrete.
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass

from .data import load_bird, build_prompt, Example
from .reward import execution_reward
from .resampling import (
    Rollout, Group, find_allwrong_groups, split_prefix_action,
    select_prefixes, resample_actions, compute_advantages,
)


@dataclass
class Cfg:
    raw: dict
    def __getitem__(self, k): return self.raw[k]


def load_cfg(path: str) -> Cfg:
    with open(path) as f:
        return Cfg(yaml.safe_load(f))


# ---- bind these to your stack ---------------------------------------------

class Policy:
    """Thin wrapper over your actor model + sampler (vLLM recommended)."""

    def generate_group(self, prompt: str, g: int, temperature: float = 1.0) -> list[Rollout]:
        # TODO: sample g rollouts; return decoded text + per-token action logprobs
        raise NotImplementedError

    def generate_from_prefix(self, prompt: str, prefix_text: str, k: int) -> list[Rollout]:
        # TODO: sample k continuations conditioned on (prompt + fixed prefix)
        raise NotImplementedError

    def ppo_update(self, advantages) -> dict:
        # TODO: PPO-clip update. Apply:
        #   L = L_clip(prefix tokens; A_prefix) + sum_k L_clip(suffix_k; A_suffix_k)
        #   plus standard GRPO loss on non-all-wrong groups.
        #   Mask prefix tokens out of the suffix advantage (mask_prefix_in_suffix_adv).
        raise NotImplementedError


# ---- training step ---------------------------------------------------------

def train_step(policy: Policy, batch: list[Example], cfg: Cfg) -> dict:
    g = cfg["grpo"]["group_size"]
    rc = cfg["resample"]

    # 1. rollout each prompt into a group, score with execution reward
    groups: list[Group] = []
    for pid, ex in enumerate(batch):
        prompt = build_prompt(ex)
        rollouts = policy.generate_group(prompt, g, cfg["grpo"]["temperature"])
        for r in rollouts:
            r.prompt_id = pid
            r.prefix_text, r.action_text = split_prefix_action(r.text)
            r.reward = execution_reward(
                r.sql, gold_rows=_gold_rows(ex), db_path=ex.db_path,
                dialect=ex.dialect,
                exec_weight=cfg["reward"]["exec_weight"],
                syntax_weight=cfg["reward"]["syntax_weight"],
                timeout_s=cfg["reward"]["exec_timeout_s"],
            )
        groups.append(Group(prompt_id=pid, rollouts=rollouts))

    # 2. EAR-SQL: resample actions for all-wrong groups, within extra budget
    resamples = []
    if rc["enabled"]:
        allwrong = find_allwrong_groups(groups)
        base_budget = g * len(batch)
        chosen = select_prefixes(allwrong, base_budget, rc["budget_ratio"], rc["K"])
        for prefix_rollout in chosen:
            ex = batch[prefix_rollout.prompt_id]
            rr = resample_actions(
                prefix_rollout, rc["K"],
                generate_from_prefix=lambda pfx, k, ex=ex: policy.generate_from_prefix(
                    build_prompt(ex), pfx, k),
                score_reward=lambda c, ex=ex: execution_reward(
                    c.sql, _gold_rows(ex), ex.db_path, ex.dialect),
            )
            resamples.append(rr)

    # 3. separated advantage streams + PPO-clip update
    adv = compute_advantages(groups, resamples)
    stats = policy.ppo_update(adv)

    # 4. metrics worth logging for the paper
    stats.update(
        all_wrong_groups=len(find_allwrong_groups(groups)),
        resampled_prefixes=len(resamples),
        recovered=sum(1 for rr in resamples if rr.recovery > 0),
    )
    return stats


def _gold_rows(ex: Example):
    from .sql_exec import execute
    res = execute(ex.db_path, ex.gold_sql, ex.dialect)
    return res.rows if res.ok else None


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    # Train on cfg.data.train (a BIRD-format split). Falls back to eval_bird only
    # if no train split is configured (skeleton default).
    train_path = cfg["data"].get("train") or cfg["data"]["eval_bird"]
    data = load_bird(train_path, cfg["data"]["db_root"], cfg["data"]["dialect"])
    print(f"[train] {len(data)} examples from {train_path}")
    # Concrete backend (vLLM rollouts + HF PPO-clip). Falls back to the stub if
    # torch/vllm/transformers aren't importable (e.g. during a dry run).
    try:
        from .policy import Policy as RealPolicy
        policy = RealPolicy(cfg.raw)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] concrete Policy unavailable ({e}); using stub. "
              "Install torch/transformers/requests + start a vLLM server.")
        policy = Policy()
    bs = cfg["grpo"]["batch_prompts"]
    sync_every = int(cfg["grpo"].get("sync_every", 0) or 0)
    for step in range(cfg["grpo"]["total_steps"]):
        batch = data[(step * bs) % len(data):][:bs]
        stats = train_step(policy, batch, cfg)
        print(step, stats, flush=True)
        # on-policy: periodically push updated actor weights into the rollout vLLM
        if sync_every and step > 0 and step % sync_every == 0 and step < cfg["grpo"]["total_steps"] - 1:
            ok = policy.sync_to_vllm() if hasattr(policy, "sync_to_vllm") else False
            print(f"[sync] step {step} actor->vLLM: {'ok' if ok else 'skip/fail'}", flush=True)

    # Save the trained actor so a fresh vLLM server can serve it for eval
    # (there is no online HF-actor -> vLLM weight sync).
    out_dir = cfg["logging"].get("out_dir")
    actor = getattr(policy, "actor", None)
    if out_dir and actor is not None:
        import os
        ckpt = os.path.join(out_dir, "actor_final")
        os.makedirs(ckpt, exist_ok=True)
        actor.save_pretrained(ckpt)
        policy.tok.save_pretrained(ckpt)
        print(f"[save] trained actor -> {ckpt}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/ear_sql.yaml")
