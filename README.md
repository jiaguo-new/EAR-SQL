# EAR-SQL — Execution-Aware Action Resampling for Text-to-SQL RL

A research code skeleton that ports the **action-resampling** idea (AXPO,
*Agent eXplorative Policy Optimization*) to **execution-reward RL for
text-to-SQL**. The goal: attack the sparse-reward bottleneck from the
**exploration side** rather than the reward-shaping side.

> Status: **skeleton / scaffolding**. The control flow, interfaces, and the
> AXPO→SQL mapping are implemented; the heavy parts (model rollout, optimizer
> step) are clearly marked `# TODO` so you can wire them to your RL stack
> (TRL / verl / OpenRLHF). It is meant to be read top-to-bottom as a spec you
> can fill in, not a turnkey trainer.

## The one idea

Execution reward is 0/1 and sparse. On hard queries an entire GRPO group is
often **all-wrong → zero advantage → no gradient**, so the hardest queries
never get learned. EAR-SQL detects those all-wrong groups, **fixes the
reasoning/schema-linking prefix**, and **resamples only the SQL action** (and
its execution-correction continuation) K times. A binary **recovery reward**
credits the prefix if any resample succeeds. This concentrates the exploration
budget exactly where the gradient is missing.

This is **orthogonal to reward shaping** (Reasoning-SQL, Progress-SQL,
Graph-Reward-SQL …) and can be stacked on top.

## Layout

```
ear_sql/
├── README.md
├── requirements.txt
├── configs/ear_sql.yaml        # all hyperparameters (K, resample budget r, etc.)
├── src/
│   ├── sql_exec.py             # execute SQL on SQLite/BigQuery, compare result sets
│   ├── reward.py               # execution + syntax reward; recovery reward
│   ├── data.py                 # BIRD / SynSQL loaders, prompt builder, M-Schema
│   ├── resampling.py           # ★ AXPO core: all-wrong detection, prefix split,
│   │                           #   uncertainty ranking, resample, advantage streams
│   └── grpo_trainer.py         # GRPO loop with the resampling hook
├── eval/
│   └── eval_bird.py            # EX + VES; optional FLEX/human-verified subset
└── scripts/
    └── train.sh
```

## Quick start (once filled in)

```bash
pip install -r requirements.txt
# 1. SFT init (Arctic-style) — out of scope here, bring your own checkpoint
# 2. RL with action resampling:
bash scripts/train.sh configs/ear_sql.yaml
# 3. Evaluate:
python eval/eval_bird.py --pred preds.jsonl --gold data/bird/dev.json --db_root data/bird/dev_databases
```

## Method ↔ code map

| Paper concept (AXPO) | text-to-SQL meaning | Where |
|---|---|---|
| Thinking prefix | reasoning + schema linking + query plan | `resampling.split_prefix_action` |
| Tool-call action | the SQL body / a DB-probe action | `resampling.split_prefix_action` |
| All-wrong tool subgroup | all rollouts in a group fail execution | `resampling.find_allwrong_groups` |
| Uncertainty ranking | mean policy prob over action tokens | `resampling.rank_prefixes_by_uncertainty` |
| Tool-call resampling | resample K SQL actions from fixed prefix | `resampling.resample_actions` |
| Recovery reward | prefix credited if any resample passes | `reward.recovery_reward` |
| Per-prefix / separated advantage | distinct advantage streams | `resampling.compute_advantages` |

## References
- AXPO — Agent eXplorative Policy Optimization for Multimodal Agentic Reasoning
- Arctic-Text2SQL-R1 (arXiv 2505.20315) — execution-reward GRPO baseline
- BIRD (bird-bench.github.io), Spider 2.0 (spider2-sql.github.io)
- Reasoning-SQL (2503.23157), Progress-SQL (2606.06825) — reward-shaping line (orthogonal)
