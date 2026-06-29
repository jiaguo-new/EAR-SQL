# On-policy RL 训练对照（Qwen2.5-Coder-7B）+ 跨基座诚实复盘

> 运行：`run_20260623_100149` ｜ 日期：2026-06-28 ｜ 分支：`claude/infallible-chatelet-127c1e`
> 模型：Qwen2.5-Coder-7B-Instruct（全量微调）｜ 单卡 GB10 ｜ 设置同 7B on-policy（30 步, sync_every=5, lr1e-5）

## 1. Coder-7B 结果（heldout60, 贪心）

| 模型 | EX | VES（全样本） |
|------|-----|---------------|
| 原始 Coder-7B（未训练） | 31.67% (19/60) | 33.24 |
| **baseline GRPO（on-policy）** | **36.67% (22/60)** | **40.83** |
| EAR-SQL（on-policy） | 33.33% (20/60) | 35.24 |

- Coder 基座更强（原始 31.67% > Qwen2.5-7B-Instruct 的 28.33%），符合预期。
- **本轮 baseline > EAR**（EX +3.34pp、VES +5.59）——与上一轮 Instruct 结论**相反**。
- 两者都比未训练有提升（on-policy 训练有效）。

## 2. ⚠️ 跨基座结论不一致 —— 现阶段 EAR vs baseline 尚无可靠信号

| 基座 | original | baseline GRPO | EAR | 谁赢 |
|------|----------|---------------|-----|------|
| Qwen2.5-7B-Instruct | 28.33% | 26.67% | **31.67%** | EAR +5.0pp |
| Qwen2.5-Coder-7B | 31.67% | **36.67%** | 33.33% | baseline +3.3pp |

**两轮方向相反。** 在 60 条留出集、单 seed 下，±3–5pp ≈ 2–3 道题，**落在噪声范围内**。
因此目前**不能下"EAR 优于/劣于 baseline"的结论**——上一轮 Instruct 的 +5pp 很可能是噪声/运气，
本轮也一样。需要扩大评测才能判定。

## 3. 一个真实的混杂因素：NaN 训练不稳定（已修）

on-policy 权重同步后，偶发 rollout 产生巨大/NaN 的重要性比，污染梯度。本轮统计：

| 训练 | NaN 步数 |
|------|----------|
| Coder baseline | 2 |
| Coder EAR | **7** |

**EAR 的 NaN 明显更多**（重采样产生更多 span，更易触发），等于 EAR 这边被不公平地多打了几次
"坏更新"。所以 Coder 这轮 baseline>EAR 也部分是 NaN 不稳定造成的，不纯是方法差异。

**已修复**（`src/policy.py` `ppo_update`）：跳过非有限的 span loss，并在 step 前检查梯度有限性，
非有限则丢弃该步更新（返回里新增 `skipped_spans`）。后续对照应更干净。

## 4. 诚实总结

- ✅ **工程**：on-policy 闭环（save+restart vLLM 同步）+ 鲁棒 SQL 抽取 + 逐条 backward +
  NaN 防护，7B/Coder-7B 全量微调单卡稳定可跑、无死机。
- ✅ **on-policy 训练确有效**：两个基座上训练后多数配置 > 未训练。
- ❓ **EAR vs baseline 未定**：两轮方向相反，受小样本噪声 + NaN 不稳定混杂。**这是当前最大的未决问题。**

## 5. 下一步（要判定 EAR 是否真有效，按顺序）

1. **用修复 NaN 后的代码重跑**，消除不公平混杂。
2. **扩评测**：全量 BIRD dev（1534）+ **多 seed（≥3）报 mean±std** + 难度分桶。
   只有这样 ±3–5pp 的差异才有统计意义。
3. 加大训练预算（steps/group/budget），看差距是否随预算稳定放大。

## 6. 产物
```
runs/op_baseline_coder/actor_final  runs/op_ear_coder/actor_final
preds/opc_{original,baseline_trained,ear_trained}.jsonl
reports/opc_eval_*.{txt,json}
configs/ear_sql_coder_op_{baseline,ear}.yaml  drive_coder_onpolicy.sh
```
