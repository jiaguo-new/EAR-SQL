# 一锤定音：EAR vs baseline GRPO 多seed对照（Qwen2.5-Coder-7B）

> 运行：`run_20260623_100149` ｜ 日期：2026-06-29 ｜ 分支：`claude/infallible-chatelet-127c1e`
> 3 seeds × {baseline, EAR}，on-policy（sync_every=5, 30 steps, lr1e-5），NaN 已修。
> 评测：dev_heldout300（无样例泄漏），贪心，VES runs=3。**全 6 个训练 0 次 NaN**（实验干净）。

## 1. 结论（统计口径）

| 模型 | EX (mean±std, n=3) | VES | per-seed EX |
|------|---------------------|-----|-------------|
| 原始 Coder-7B（未训练） | 38.00% | 40.00 | — |
| baseline GRPO | **40.00 ± 0.58%** | 40.92 | 39.67 / 39.67 / 40.67 |
| EAR-SQL | **39.78 ± 1.90%** | 41.44 | 40.33 / 41.33 / 37.67 |

**VERDICT：EAR 与 baseline 统计上打平。** 差异 −0.22pp，|Δ|/pooled_std = **0.16**
（要 >~2 才能算真效应）。**当前预算下 EAR 不能可靠地优于普通 GRPO。**

- ✅ **on-policy 训练有效**：baseline 与 EAR 都比未训练高（38.0 → ~40%）。
- ❌ **EAR 没有可靠增益**：均值几乎相同，且 **EAR 方差更大**（1.90 vs 0.58），更不稳定
  （seed3 EAR 退化到 37.67%，模型变啰嗦）。
- 之前 60 条单 seed 的"EAR +5pp"(Instruct) 与"baseline +3.3pp"(Coder) **都是噪声**——
  本轮 300×3seed 把它们一并证伪。这正是做严谨评测的价值。

## 2. 难度分桶（EX%）

| 难度 (n) | original | baseline | EAR |
|----------|----------|----------|-----|
| simple (194) | 43.3 | 44.8±1.0 | 45.7±2.1 |
| moderate (79) | 29.1 | 32.9±1.3 | 30.0±1.9 |
| challenging (27) | 25.9 | 25.9±0.0 | 25.9±0.0 |

- 训练增益主要在 simple/moderate；**challenging 三者都没动（25.9%）**。
- EAR 本应在"全错难题"上发力，但**最难的 challenging 桶毫无改善**——说明当前 K=2/小预算下，
  动作重采样还救不动真正的硬题。这是 EAR 想证明价值的关键短板。

## 3. 为什么是这个结果（机制层面）

- baseline 在全错组上无梯度（updated_spans=0 的空步），EAR 把它们转成后缀重采样更新——
  机制确实在跑（每轮数十次重采样），但**重采样后缀大多仍错**（救回率极低），
  所以没转化成留出集 EX 的稳定提升。
- 与历史一致：frozen/弱策略下"全错组转化率"本就很低；要让 EAR 起效，可能需要
  更大 K/探索预算、或在更强策略上、或针对 challenging 的更强 schema-linking。

## 4. 诚实总结

**这是一个干净的 null 结果**：在 Qwen2.5-Coder-7B + 300×3seed + on-policy 30步 的严谨设置下，
**EAR-SQL 与 baseline GRPO 无显著差异**。on-policy 训练本身有效（+2pp over 未训练），
但 EAR 的"分离优势流 + 动作重采样"在此预算下没带来可复现的额外收益，且更不稳定。

## 5. 要让 EAR 真正显出价值，可试方向

1. **加大重采样预算**：K=4~8、budget_ratio 更高，给难题更多翻盘机会（直接攻 challenging 桶）。
2. **更长训练 + 更大 group_size**，让全错组转化率有机会累积成 EX 增益。
3. **针对 challenging 的 schema-linking / evidence 强化**（当前 challenging 全程 25.9%，是天花板瓶颈）。
4. 全量 dev 复核 + 更多 seed，缩小置信区间。

## 6. 产物
```
sweep_coder/{baseline,ear}_s{1,2,3}/actor_final
preds/sw_*.jsonl  reports/sw_eval_*.{txt,json}
reports/sweep_coder_summary.txt   # 聚合输出
configs/ear_sql_sweep_{baseline,ear}.yaml  scripts/aggregate_sweep.py
drive_sweep_coder.sh
```
