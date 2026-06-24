# 基线 GRPO vs EAR-SQL 对照报告（BIRD dev 前 100 条）

> 运行：`run_20260623_100149` ｜ 日期：2026-06-23 ｜ 分支：`claude/infallible-chatelet-127c1e`
> 机器：thinkstationpgx-31d2 / NVIDIA GB10（统一内存 119GB，aarch64）
> 推理后端：vLLM 0.17.1（conda `vllm-cuda`，torch 2.10.0+cu130）

## 1. 实验设置

由于 14B 单卡在 GB10 上无法在预算内完成全量 RL 训练（详见
`run_20260621_005814/reports/experiment_summary.md` 的吞吐分析），本轮沿用
**推理时 best-of-N** 作为基线 vs EAR-SQL 的可复现对照：

- **BASE_MODEL**：`ckpts/sft_init_sqlplus_merged`（Qwen3-14B + sqlplus LoRA 合并，
  **未在 BIRD dev 上训练**，避免评测泄漏）。
- **基线**：每题采样 G=8 条 rollout，取执行奖励最高者为预测。
- **EAR-SQL**：在同一组 base rollout 上，对**全错组**按动作不确定性选前缀，条件重采样
  K=2 条后缀（budget_ratio=0.5），最终从 base + 重采样中取最优。
- **关键改进（本轮）**：baseline 与 EAR **共享同一组 base rollout**
  （`gen_preds_bestofn.py --ear --baseline-out`），因此两者差异**纯粹来自动作重采样**，
  消除了上一轮分别采样（temperature=1.0）带来的随机噪声。
- 样本：BIRD dev 前 100 条（上一轮仅 20 条）。
- 采样：temperature=1.0，seed=42，max_gen_len=1024，chat template **关闭思考模式**
  （`enable_thinking=False`，见下「修复记录」）。

## 2. 主对照结果

| 指标 | 基线 (best-of-8) | EAR-SQL | 差异 |
|------|------------------|---------|------|
| **Execution Match (EX)** | 48 / 100 = **48.0%** | 49 / 100 = **49.0%** | **+1.0 pp** |
| **VES（全样本, 0-100）** | **48.36** | **49.03** | **+0.67** |
| VES（仅 EX 正确子集） | 100.74 | 100.06 | −0.68 |
| Valid SQL Rate | 76 / 100 = 76.0% | 76 / 100 = 76.0% | +0.0 pp |
| JOIN EX | 33 / 71 = 46.48% | 34 / 71 = 47.89% | +1.41 pp |
| Exact Match (EM) | 1 / 100 = 1.0% | 1 / 100 = 1.0% | +0.0 pp |

- **EAR 严格不劣于基线**：逐题对比，EAR 相比基线**新增正确 1 题（idx=88），无任何回退**。
- 该 +1 EX 正是被动作重采样「救回」的那个全错组。
- **VES**（Valid Efficiency Score，`eval/eval_bird.py --ves-runs 5` 实现）：
  - 全样本 VES 随 EX 走（错的样本效率贡献 0），EAR 因多救回 1 题而略高（49.03 vs 48.36）。
  - 仅看 EX 正确子集，VES≈100，说明预测 SQL 的执行耗时与 gold 基本持平（同库、同量级查询）；
    EAR 子集略低于基线（100.06 vs 100.74），因为新救回的那题是个较慢的难查询，拉低了均值。
  - VES 计算：对每个 EX 通过的样本 `R=sqrt(t_gold/t_pred)`，5 次执行取均值降噪，错的样本 R=0，
    `VES = 100/N · ΣR`。

## 3. 全错组转化率（EAR 核心证据）

| 量 | 值 |
|----|----|
| 全错组数 `all_wrong_groups` | **53 / 100**（53%）|
| 触发重采样的前缀数 `resampled_prefixes` | 53 |
| 被救回数 `recovered` | **1** |
| **全错组转化率 = recovered / all_wrong** | **1 / 53 ≈ 1.9%** |

**解读：**

- 全错组占比高达 53%，说明 frozen SFT-init 策略在 BIRD 上有大量「8 条 rollout 全错」的难题。
- 动作重采样管线对全部 53 个全错组都成功触发（resampled=53），证明
  「全错检测 → 按动作不确定性选前缀 → 条件重采样后缀」全链路在真机上可用。
- 但**转化率仅 1.9%**：在 frozen SFT-init 策略 + K=2 小预算下，重采样后缀大多仍错。
  这与上一轮（20 条，转化 2/7≈29%，但样本极小）一起印证一个结论：
  **动作重采样的收益受策略本身上限制约**——要稳定提高转化率，需要
  (a) 经 RL 训练后的策略来生成后缀（而非 frozen SFT-init），和/或 (b) 更大的 K / 探索预算。

## 4. 与上一轮（20 条）对比

| 轮次 | 样本 | baseline EX | EAR EX | 全错组转化率 | 备注 |
|------|------|-------------|--------|--------------|------|
| run_20260621（旧） | 20 | 65.0% | 75.0% | 2/7 ≈ 29% | baseline/EAR 分别采样，含思考模式截断噪声 |
| **run_20260623（本轮）** | **100** | **48.0%** | **49.0%** | **1/53 ≈ 1.9%** | 共享 base rollout，关闭思考模式，更干净更大样本 |

本轮 EX 绝对值低于旧轮，主要因为：样本扩大到 100（覆盖更多难题），且采用共享 rollout
的严格对照。本轮结论更可信：**EAR 在干净对照下给出小幅正收益（+1pp），无回退**，
但单纯推理时重采样、配 frozen 策略，收益有限。

## 5. 修复记录（遇错自修）

1. **思考模式截断**：初测发现 Qwen3 chat template 默认开启 thinking，1024-token 预算被
   `<think>` 推理耗尽，模型从不输出 `<sql>…</sql>`，导致预测被截断为推理散文、EX 虚低、
   全错组虚高。**修复**：在 `scripts/gen_preds_bestofn.py` 的 chat 模板调用中加
   `enable_thinking=False`。效果：5 条样例耗时 5min→1.5min（3.3×），预测全部变为合法 SQL，
   全错组 4/5→2/5。（上一轮 20 条对照同样受此 bug 影响。）
2. **环境**：未使用 bootstrap 脚本的 `python -m venv + pip install`（aarch64/GB10 上会装错
   torch/vllm 轮子），改用现成 conda 环境 `vllm-cuda`（已验证 vLLM 0.17.1 可跑 14B）。
3. **资源冲突**：端口 8000 原有一个空转 14.5h 的 Qwen3-8B vLLM 服务占 54GB，与 14B 无法共存；
   经用户授权后停止，释放内存启动 14B（gpu_memory_utilization=0.45）。

## 6. 产物

```
run_20260623_100149/
├── reports/machine.txt
├── reports/eval_baseline_100.{txt,json}   # EX 48%
├── reports/eval_ear_100.{txt,json}        # EX 49%
├── reports/baseline_100_metrics.json      # all_wrong=0
├── reports/ear_100_metrics.json           # all_wrong=53, recovered=1
├── preds/baseline_100.jsonl
├── preds/ear_100.jsonl
└── logs/{smoke.log, vllm_14b.log, gen_100.log}
```

## 7. 结论与下一步

- **跑通**：smoke 全绿、14B 单卡 vLLM rollout、BIRD 数据、best-of-N 基线/EAR、EX 评测、
  全错组转化率统计——全链路在 GB10 上闭环。
- **结论**：干净对照下 EAR-SQL 相比基线 EX +1pp 且零回退，但推理时 + frozen SFT-init 策略下
  全错组转化率低（1.9%）。EAR 的真正收益需配合 RL 训练后的策略。
- **下一步**：
  1. 补全 `src/policy.py` 标准 GRPO loss（当前非全错组更新为 0），在更强算力上跑真 RL 训练；
  2. ~~在 `eval/eval_bird.py` 加入 VES~~ ✅ 已完成（`--ves-runs`，见上表）；
  3. 扩大样本到全量 1534 条并按 BIRD 难度分桶看增益分布。
