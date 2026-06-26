# 真 RL 训练闭环：baseline GRPO vs EAR-SQL（Qwen3-1.7B）

> 运行：`run_20260623_100149` ｜ 日期：2026-06-26 ｜ 分支：`claude/infallible-chatelet-127c1e`
> 机器：NVIDIA GB10（119GB 统一内存）｜ vLLM 0.17.1 / torch 2.10 / conda `vllm-cuda`

## 1. 这次验证了什么

首次在本机端到端跑通 **训练→存盘→reload→评测** 的闭环，并完成 **baseline GRPO vs
EAR-SQL 的真 RL 训练对照**（此前被算力 + 死机阻塞）：

- 数据无泄漏：BIRD dev 切分为 `dev_train180`（训练）/ `dev_heldout60`（评测），
  同 11 库、零样例重叠（seed=42）。
- 三个模型在 **同一留出集、同一贪心解码** 下对比，隔离"训练"本身的效果。
- 训练侧 baseline 关重采样、EAR 开重采样，其余超参完全一致
  （group_size=4, batch_prompts=2, lr=1e-5, 16 steps）。

## 2. 结果（heldout60，贪心单样本）

| 模型 | EX | VES（全样本） | VES（EX 正确子集） |
|------|-----|---------------|---------------------|
| 原始 1.7B（未训练） | **18.33%** (11/60) | 19.93 | 108.74 |
| baseline 训练（GRPO，关重采样） | 15.00% (9/60) | 17.28 | 115.20 |
| **EAR 训练（GRPO+动作重采样）** | **16.67%** (10/60) | **19.38** | 116.30 |

**关键对照（实验主结论）：在完全相同的训练条件下，EAR > baseline：EX +1.67pp、
VES +2.10。** 这与论文主张（前缀/后缀分离优势流的 GRPO 更新优于普通 GRPO）方向一致。

## 3. 诚实的局限（结果为何偏弱）

- **两者都低于未训练原始模型**。原因是训练预算极小且为 off-policy：
  - rollouts 来自 vLLM 的**冻结初始权重**，actor 离线更新（skeleton 架构无在线权重同步）；
  - 仅 16 步、lr=1e-5、1.7B 小模型——不足以让弱基座产生正向收敛，主要表现为噪声/轻微退化。
- 留出集仅 60 条，±2pp 在噪声范围内；EAR>baseline 是方向性证据，非统计显著。
- 训练期 EAR 指标：每步检测到 1–2 个全错组并触发重采样，`recovered` 多为 0
  （frozen rollout 策略下难题难救回，与推理时结论一致）。

## 4. 工程修复（让闭环可跑、不死机）

1. **死机根因修复**：`src/policy.py` `ppo_update` 原本把整 batch 所有 rollout 的前向累成
   一个 autograd 大图再一次性 backward → 激活内存 ×N，8B 全量微调时撑爆 GB10 统一内存、
   **整机卡死并重启**。改为**逐条 rollout 单独 backward 累加梯度**（峰值激活=1条）。
2. **空生成保护**：vLLM 偶尔返回空串，tokenizer 得到 0 长度序列导致 actor 前向崩溃；
   在 `ppo_update` 中跳过空 span。
3. **trainer**：从 `cfg.data.train` 训练（而非 eval 集）+ 训练后保存 actor 到
   `out_dir/actor_final`，供新 vLLM 加载评测。
4. **m_schema**：每列示例值 3→1，避免个别 BIRD 库 prompt 涨到 26k token 超上下文。
5. **显存编排**：1.7B 全量微调 ≈ 10GB + vLLM@0.30 ≈ 全程峰值 ~60GB / 119GB，零 swap、零死机。

> ⚠️ 经验教训：**8B 全量微调 + vLLM 在单卡 GB10（统一内存）上会卡死整机**。本机真训练
> 应使用 ≤1.7B 全量微调，或 LoRA，或多卡/大显存环境。

## 5. 产物

```
run_20260623_100149/
├── runs/train_baseline_1p7b/actor_final/   # 训练后 checkpoint
├── runs/train_ear_1p7b/actor_final/
├── preds/c17_{original,baseline_trained,ear_trained}.jsonl
├── reports/c17_eval_*.{txt,json}
└── logs/{train_baseline_1p7b,train_ear_1p7b,drive_1p7b}.log
configs/ear_sql_1p7b_{baseline,ear}.yaml
```

## 6. 下一步（要拿到"训练后超过未训练"的正收益）

1. **更大训练预算**：数百步、更合适的 lr，并解决 off-policy（周期性把 actor 权重同步回
   vLLM rollout 服务，做真正的 on-policy GRPO）。
2. **更强基座**：4B/7B（需多卡或大显存以避免 GB10 死机），1.7B 上限太低。
3. 扩留出集到全量 dev 并按难度分桶，给 EAR vs baseline 的 delta 做统计显著性。
