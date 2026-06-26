# 真 RL 训练闭环（7B）：baseline GRPO vs EAR-SQL

> 运行：`run_20260623_100149` ｜ 日期：2026-06-26 ｜ 分支：`claude/infallible-chatelet-127c1e`
> 模型：Qwen2.5-7B-Instruct（全量微调）｜ 机器：NVIDIA GB10（119GB 统一内存）

## 1. 这次主要解决了什么（工程）

之前 8B 全量微调把 GB10 **整机卡死重启**。根因已修（`src/policy.py` 的 `ppo_update`
原本把整 batch 所有 rollout 累成一个 autograd 大图再一次性 backward，激活内存 ×N）：
改为**逐条 rollout 单独 backward 累加梯度**，峰值激活=1条。

修复后 **7B 全量微调真训练闭环可安全运行**（两条路径都验证过）：
- **单机**（本次采用）：本地 vLLM @util 0.15 与 7B actor 共存，峰值 ~89GB/119GB，swap≤2.7GB，**无死机**。
- **双机**：一台 GB10 跑 vLLM rollout、另一台跑 actor（200G 互联，626MB/s 拷模型）。
  本次因对端 GPU 被另一工作占满（96%）rollout 仅 ~15tok/s，遂改单机。

## 2. 结果（dev_heldout60，贪心，无泄漏切分）

| 模型 | EX | VES（全样本） | VES（EX 正确子集） |
|------|-----|---------------|---------------------|
| 原始 Qwen2.5-7B-Instruct | 10.0% (6/60) | 8.94 | 89.43 |
| baseline 训练（GRPO，关重采样） | 10.0% (6/60) | 10.15 | 101.53 |
| EAR 训练（GRPO+动作重采样） | 10.0% (6/60) | 9.82 | 98.24 |

训练设置：group_size=4, batch_prompts=4, lr=1e-5, 16 steps；EAR 共检测 26 个全错组、
触发 26 次重采样、救回 1 次。

## 3. 诚实结论：本预算下不显著

- **EX 三者持平（都 10.0%）**：16 步、lr=1e-5 不足以移动 EX。
- **VES 差异在噪声内**（±1）：baseline-trained 10.15 ≈ EAR-trained 9.82 ≈ 原始 8.94。
- 与 1.7B 那轮对比：1.7B 上 EAR>baseline，7B 上 baseline≥EAR——**方向相反**，进一步说明
  当前预算下 EAR vs baseline 的差异是噪声，不是信号。
- **Qwen2.5-7B-Instruct 基线异常低(10%)**：同设置下 Qwen3-8B 贪心是 36.67%。怀疑是
  prompt/chat-template 适配问题（`enable_thinking=False` 是 Qwen3 专有 kwarg，对 Qwen2.5
  可能未理想生效），需排查——这会压低三者的绝对值与可分辨度。

## 4. 为什么还没看到正收益（根本限制，未变）

1. **off-policy**：rollouts 来自 vLLM 冻结权重，actor 离线更新、训练中不回灌 vLLM；
   真正 on-policy 需周期性把 actor 权重同步回 rollout 服务。
2. **预算太小**：16 步 + lr1e-5 + 60 条留出集。要出统计显著的 delta 需数百步、更大留出集、多 seed。
3. **基线模型选择**：Qwen2.5-7B-Instruct 在本 prompt 下表现差；建议换 Qwen2.5-Coder-7B
   （下载中，~18h）或排查 Qwen2.5 的 chat-template。

## 5. 产物

```
run_20260623_100149/
├── runs/train_baseline_7b/actor_final/   # 训练后 checkpoint
├── runs/train_ear_7b/actor_final/
├── preds/sn7_{original,baseline_trained,ear_trained}.jsonl
├── reports/sn7_eval_*.{txt,json}
├── drive_7b_singlenode.sh                # 单机闭环驱动
└── drive_7b_distributed.sh               # 双机版（对端空闲时可用）
configs/ear_sql_7b_{baseline,ear}.yaml
```

## 6. 下一步建议（按价值）

1. **排查/修正 Qwen2.5 的 prompt 适配**（先把原始 EX 拉回合理区间，否则训练对照没意义）。
2. **加 on-policy 权重同步 + 数百步训练**（这是看到"训练后 > 原始"的关键）。
3. 待 **Qwen2.5-Coder-7B** 下完用它复跑（coder 基座在 SQL 上更强）。
4. 留出集扩到全量 dev + 难度分桶 + 多 seed，做统计显著性。
