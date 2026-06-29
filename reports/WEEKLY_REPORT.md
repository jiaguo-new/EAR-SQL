# EAR-SQL 周报

> 周期：2026-06-23 ~ 2026-06-29 ｜ 分支：`claude/infallible-chatelet-127c1e`
> 机器：双 NVIDIA GB10（统一内存 119GB/台，200G 互联）｜ 仓库：jiaguo-new/EAR-SQL

## 一、本周目标

验证 EAR-SQL（在 GRPO 上对"全错组"做动作重采样 + 前缀/后缀分离优势流）相对普通
GRPO 是否有真实增益，从"推理时对照"一路推进到"真 RL 训练 + 多 seed 严谨对照"。

## 二、本周完成（按主线）

### 1. 打通评测与推理对照
- 跑通环境/smoke/14B vLLM rollout/BIRD 数据，产出 baseline vs EAR 的 best-of-N 推理对照。
- **实现 VES 指标**（`eval/eval_bird.py --ves-runs`）：补齐 EX/VES 双指标（原仅 EX）。

### 2. 真 RL 训练闭环（从 0 到可跑）
- 搭建 **train → 存 checkpoint → reload vLLM → 留出集评测** 完整闭环。
- 数据无泄漏：BIRD dev 切分 train/held-out（同库、零样例重叠）。
- 规模演进：1.7B（验证闭环）→ 7B → Qwen2.5-Coder-7B（SQL 专用基座）。

### 3. 三个关键工程修复（缺一闭环不成立）
| 问题 | 现象 | 修复 |
|------|------|------|
| **整机死机** | 8B 全量微调把 119GB 统一内存挤爆、重启 | `ppo_update` 改逐条 rollout backward（峰值激活=1条），单卡 7B 全量微调峰值 ~90GB、无死机 |
| **EX 被严重低估** | 模型不打 `<sql>` 标签时抽取取了末行散文 | 鲁棒抽取（`<sql>`→```sql→末条 SELECT/WITH→末行），7B 原始 EX 10%→28% |
| **on-policy 缺失** | rollout 来自冻结权重，训练不动 EX | 每 N 步 actor→vLLM 权重同步（save+restart） |
| **NaN 不稳定** | on-policy 后偶发巨大重要性比 NaN 掉模型 | 跳过非有限 loss/梯度；6 个训练全程 0 NaN |

### 4. 多 seed 严谨对照（一锤定音）
- Qwen2.5-Coder-7B，on-policy，留出集 300 条，**3 seeds**，难度分桶，全程 0 NaN。

## 三、关键结果

### 最终判定（Coder-7B，heldout300，3 seeds，mean±std）
| 模型 | EX | VES |
|------|-----|-----|
| 原始（未训练） | 38.00% | 40.00 |
| baseline GRPO | **40.00 ± 0.58%** | 40.92 |
| EAR-SQL | **39.78 ± 1.90%** | 41.44 |

**结论：EAR 与 baseline 统计上打平**（Δ=−0.22pp，|Δ|/pooled_std=0.16，需 >2 才显著）。
- ✅ on-policy 训练有效：两者均 +2pp over 未训练。
- ❌ EAR 此预算下无可靠增益，且方差更大、更不稳定。
- 早期 60 条单 seed 的"EAR +5pp / baseline +3.3pp"经查**均为噪声**——严谨评测的价值所在。

### 难度分桶（EX%）
| 难度(n) | original | baseline | EAR |
|---------|----------|----------|-----|
| simple(194) | 43.3 | 44.8 | 45.7 |
| moderate(79) | 29.1 | 32.9 | 30.0 |
| challenging(27) | 25.9 | 25.9 | **25.9** |

→ 增益集中在 simple/moderate；**最难 challenging 桶三者纹丝不动**——EAR 的重采样在
K=2/小预算下救不动硬题，这是其价值未显现的核心瓶颈。

## 四、问题与阻塞

1. **下载慢**：modelscope ~200–400kB/s（不走代理也一样），Coder-7B（15GB）约 18h 才下完。
2. **对端 GB10 被占**：另一工作（qwen3-14b-cdc）占满 GPU，分布式 rollout 仅 ~15tok/s，故改单机。
3. **eval 偏慢**：训练后模型生成偏啰嗦，单次 300 条评测 ~40min，全 sweep ~12h（隔夜跑完）。
4. git push 需经 `GIT_SSH_COMMAND` 指定 key 绕过本地代理拦截（已记录）。

## 五、本周结论

- **工程上**：在单卡 GB10 上把"7B 真 on-policy RL 训练闭环"做到稳定可跑、可复现、不死机。
- **科学上**：得到一个**干净的 null 结果**——EAR-SQL 在当前规模/预算下相对 baseline GRPO
  无显著优势。机制能运行（每轮数十次重采样），但难题救回率太低，未转化为 EX 增益。

## 六、下周计划

1. **加大重采样预算**（K=4~8、budget_ratio↑），直接冲 challenging 桶——最可能让 EAR 翻盘。
2. **更长训练 + 更大 group_size**，让"全错组转化率"累积成 EX 增益。
3. **challenging 难题的 schema-linking / evidence 强化**（当前 25.9% 是天花板瓶颈）。
4. 全量 dev 复核 + 更多 seed，缩小置信区间。

## 附：关键产物
```
reports/sweep_coder_verdict.md        # 最终判定（含统计与分桶）
reports/sweep_coder_summary.txt       # 聚合输出
reports/train_{7b,coder}_onpolicy.md  # 各阶段过程报告
eval/eval_bird.py (VES) | src/policy.py (微批/同步/NaN防护) | scripts/aggregate_sweep.py
drive_*.sh（单机/分布式/多seed 驱动）
```
