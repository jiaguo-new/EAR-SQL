# On-policy RL 训练对照（7B）：EAR-SQL 首次跑出正收益

> ⚠️ **更新（2026-06-28）：本轮 EAR>baseline 的结论未在 Qwen2.5-Coder-7B 上复现**
> （Coder 上 baseline 36.67% > EAR 33.33%，方向相反）。在 60 条单 seed 下 ±5pp 属噪声，
> 且训练有 NaN 不稳定混杂。**当前 EAR vs baseline 尚无可靠结论**，详见
> `train_coder_onpolicy.md`。本文结果请据此谨慎看待。

> 运行：`run_20260623_100149` ｜ 日期：2026-06-27 ｜ 分支：`claude/infallible-chatelet-127c1e`
> 模型：Qwen2.5-7B-Instruct（全量微调）｜ 机器：NVIDIA GB10（单卡，119GB 统一内存）

## 1. 头条结果

留出集 dev_heldout60（与训练集无样例重叠），贪心解码：

| 模型 | EX | VES（全样本） | VES（EX 正确子集） |
|------|-----|---------------|---------------------|
| 原始 Qwen2.5-7B-Instruct（未训练） | 28.33% (17/60) | 28.48 | 100.50 |
| baseline GRPO（on-policy） | 26.67% (16/60) | 27.67 | 103.78 |
| **EAR-SQL（on-policy）** | **31.67% (19/60)** | **30.22** | 95.44 |

**关键结论：**
- **EAR 是唯一超过未训练模型的配置**：EX +3.34pp（28.33→31.67）、VES +1.74。
- **EAR 显著优于 baseline GRPO**：EX **+5.0pp**（26.67→31.67）、VES +2.55。
- **普通 GRPO（baseline）反而略降**（EX 28.33→26.67）：在大量"全错组"上无可用学习信号。

这正是 EAR-SQL 的论点：对 8 条 rollout 全错的难题，普通 GRPO 没有正样本、无法更新；
EAR 通过"按动作不确定性选前缀 + 条件重采样后缀 + 分离优势流"把这些难题转化为学习信号。

## 2. 让结果成立的三处关键改动（本轮新增）

1. **on-policy 权重同步**：每 5 步把 actor 权重存盘并重启 rollout vLLM，使后续 rollout 来自
   当前策略（`grpo.sync_every`，`scripts/restart_rollout_vllm.sh`）。这是从"训练无效"到
   "训练有效"的关键——此前 off-policy（rollout 永远来自冻结初始权重）下训练不动 EX。
2. **鲁棒 SQL 抽取**：`<sql>`标签 → ```sql 围栏 → 最后一条 SELECT/WITH 语句 → 末行。
   修复前模型不打`<sql>`标签时会把尾部解释散文当成预测，原始 7B EX 被压到 10%；修复后 28.33%。
   该抽取同时用于**训练奖励**（`policy._extract_sql`），让奖励信号也变干净。
3. **逐条 backward（防死机）**：`ppo_update` 不再把整 batch 的 rollout 累成一个大图，
   峰值激活=1条，7B 全量微调 + 本地 vLLM 单卡共存峰值 ~90GB/119GB、无死机。

## 3. 机制证据（训练过程）

| 训练 | 总重采样前缀 | 训练期救回 | updated_spans=0 的步（全错组被浪费） |
|------|--------------|------------|--------------------------------------|
| baseline GRPO | 0 | 0 | 有（全错步无信号、空更新） |
| EAR | **59** | 1 | 无（全错组经重采样仍产生更新） |

baseline 在全错步上 `updated_spans=0`（白跑）；EAR 把同样的全错组转成 59 次后缀重采样更新——
这就是 EAR 多出来那 +5pp 的来源。

## 4. 训练设置

- group_size=4, batch_prompts=4, lr=1e-5, total_steps=30, sync_every=5（共 5 次权重同步）。
- 数据：dev_train180 训练 / dev_heldout60 评测（seed=42 切分，零重叠，同 11 库）。
- 单卡 GB10：本地 vLLM @util 0.15 与 7B actor 共存；评测时 reload vLLM @0.35（无 actor）。
- 每个训练 run ~70–80min，全闭环 ~2.7h。

## 5. 局限（仍需注意）

- **样本小**：留出集 60 条，+5pp ≈ 3 道题，单 seed。方向性强（EAR 唯一改善、baseline 退化），
  但要统计显著需扩到全量 dev + 多 seed。
- **绝对值仍低**：Qwen2.5-7B-Instruct 基座在 BIRD 上偏弱（28%）；换 Qwen2.5-Coder-7B
  （下载中）或更大基座应更高。
- 30 步仍是小预算；更多步 + 更大 group/budget 预期进一步拉开 EAR 与 baseline。

## 6. 产物

```
run_20260623_100149/
├── runs/op_baseline_7b/actor_final/   # on-policy baseline ckpt
├── runs/op_ear_7b/actor_final/        # on-policy EAR ckpt
├── preds/op7_{original,baseline_trained,ear_trained}.jsonl
├── reports/op7_eval_*.{txt,json}
├── drive_7b_onpolicy.sh
└── logs/train_op_{baseline,ear}_7b.log
configs/ear_sql_7b_op_{baseline,ear}.yaml
scripts/restart_rollout_vllm.sh
```

## 7. 下一步

1. 扩留出集到全量 dev + 多 seed + 难度分桶 → 给 +5pp 做统计显著性与"难题增益集中度"。
2. Qwen2.5-Coder-7B 下完复跑（更强 SQL 基座）。
3. 加大 total_steps / group_size / budget_ratio，看 EAR vs baseline 差距是否随预算扩大。
