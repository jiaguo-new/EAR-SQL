"""EAR-SQL core: action resampling for all-wrong GRPO groups.

This is the AXPO mechanism ported to text-to-SQL. A "group" is the set of G
rollouts sampled for one prompt. Each rollout is a trajectory:

    prefix  = reasoning + schema linking + query plan   (the "thinking")
    action  = the SQL body / a DB-probe step            (the "tool call")
    suffix  = action + any execution-correction continuation

When the whole group is all-wrong, we fix the prefix and resample only the
action+suffix, concentrating exploration where the gradient is missing.

The functions below are framework-agnostic dataclasses + pure logic; the actual
token generation and logprob computation are injected as callables so you can
back them with TRL / vLLM / verl.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


# ---- trajectory containers -------------------------------------------------

@dataclass
class Rollout:
    prompt_id: int
    text: str                      # full decoded trajectory
    prefix_text: str               # reasoning + schema linking + plan
    action_text: str               # the SQL action (and continuation)
    action_token_logprobs: list[float]  # policy logprobs over action tokens
    sql: str                       # extracted final SQL
    reward: float = 0.0            # execution reward (filled by trainer)


@dataclass
class Group:
    prompt_id: int
    rollouts: list[Rollout]

    @property
    def all_wrong(self) -> bool:
        # "all-wrong tool-using subgroup": every action-taking rollout failed.
        return all(r.reward <= 0 for r in self.rollouts)


@dataclass
class ResampleResult:
    prompt_id: int
    prefix_text: str
    resampled: list[Rollout]
    recovery: float = 0.0          # binary recovery reward for the prefix


# ---- 1. detect all-wrong groups -------------------------------------------

def find_allwrong_groups(groups: list[Group]) -> list[Group]:
    """Groups where execution reward is uniformly zero -> zero GRPO advantage."""
    return [g for g in groups if g.all_wrong]


# ---- 2. split prefix / action ---------------------------------------------

def split_prefix_action(text: str, sql_open: str = "<sql>") -> tuple[str, str]:
    """Cut a trajectory at the first action boundary.

    Here the boundary is the start of the SQL emission (`<sql>`). For an agentic
    setup, cut at the first <tool_call> instead. Everything before is the fixed
    prefix; everything from the boundary on is resampled.
    """
    idx = text.find(sql_open)
    if idx == -1:
        return text, ""          # no action found; treat whole thing as prefix
    return text[:idx], text[idx:]


# ---- 3. uncertainty ranking -----------------------------------------------

def action_uncertainty(r: Rollout) -> float:
    """Mean policy probability over action tokens; lower = more uncertain.

    AXPO prioritizes the *least confident* prefixes for resampling, since those
    are where focused exploration pays off most.
    """
    if not r.action_token_logprobs:
        return 1.0
    return float(np.exp(np.mean(r.action_token_logprobs)))


def rank_prefixes_by_uncertainty(group: Group) -> list[Rollout]:
    # representative rollout per distinct prefix, sorted by ascending confidence
    return sorted(group.rollouts, key=action_uncertainty)


# ---- 4. budgeted selection -------------------------------------------------

def select_prefixes(allwrong: list[Group], base_budget: int, budget_ratio: float,
                    K: int) -> list[Rollout]:
    """Breadth-first pick of the most-uncertain prefixes within the extra budget.

    base_budget = G * num_prompts (the normal rollout count this step).
    Extra resampling is capped at budget_ratio * base_budget.
    """
    max_extra = int(budget_ratio * base_budget)
    n_prefixes = max(0, max_extra // max(K, 1))
    ranked: list[Rollout] = []
    for g in allwrong:
        ranked.extend(rank_prefixes_by_uncertainty(g)[:1])  # one prefix per group
    ranked.sort(key=action_uncertainty)
    return ranked[:n_prefixes]


# ---- 5. resample actions ---------------------------------------------------

def resample_actions(prefix_rollout: Rollout, K: int,
                     generate_from_prefix: Callable[[str, int], list[Rollout]],
                     score_reward: Callable[[Rollout], float]) -> ResampleResult:
    """Fix the prefix, generate K new action+continuations, score each.

    `generate_from_prefix(prefix_text, K)` must sample K continuations conditioned
    on the (prompt + fixed prefix) and return Rollouts with the *new* action and
    its logprobs. `score_reward` runs SQL and returns execution reward.
    """
    cands = generate_from_prefix(prefix_rollout.prefix_text, K)
    for c in cands:
        c.reward = score_reward(c)
    from .reward import recovery_reward
    rec = recovery_reward([c.reward for c in cands])
    return ResampleResult(
        prompt_id=prefix_rollout.prompt_id,
        prefix_text=prefix_rollout.prefix_text,
        resampled=cands,
        recovery=rec,
    )


# ---- 6. advantage computation (separated streams) --------------------------

def _grpo_norm(rewards: list[float]) -> list[float]:
    a = np.asarray(rewards, dtype=np.float64)
    mu, sd = a.mean(), a.std()
    if sd < 1e-8:
        return [0.0] * len(rewards)
    return ((a - mu) / sd).tolist()


@dataclass
class AdvantageBundle:
    # advantages applied to ORIGINAL group rollouts (standard GRPO when not all-wrong)
    base: dict[int, list[float]] = field(default_factory=dict)
    # per-prefix advantages for resampled suffixes (suffix tokens only)
    suffix: list[tuple[ResampleResult, list[float]]] = field(default_factory=list)
    # per-prefix advantage for the source prefix tokens, from recovery reward
    prefix: list[tuple[ResampleResult, float]] = field(default_factory=list)


def compute_advantages(groups: list[Group], resamples: list[ResampleResult]) -> AdvantageBundle:
    """Separate advantage streams to avoid conflicting signals on prefix tokens.

    - Non-all-wrong groups: standard GRPO over execution reward.
    - Resampled suffixes: per-prefix GRPO over the K resample rewards
      (applied to suffix tokens only; prefix masked).
    - Source prefix tokens: updated by the binary recovery reward, GRPO-normalized
      across the selected prefixes in the step.
    """
    bundle = AdvantageBundle()

    for g in groups:
        if not g.all_wrong:
            bundle.base[g.prompt_id] = _grpo_norm([r.reward for r in g.rollouts])

    for rr in resamples:
        bundle.suffix.append((rr, _grpo_norm([c.reward for c in rr.resampled])))

    if resamples:
        rec_norm = _grpo_norm([rr.recovery for rr in resamples])
        for rr, adv in zip(resamples, rec_norm):
            bundle.prefix.append((rr, adv))

    return bundle
