# EAR-SQL 算力机操作手册（中文）

> 目标：在算力机 `dameng@192.168.1.141` 上新建实验目录，跑通 EAR-SQL 实验路线，并产出 EX/VES 评测。
>
> ⚠️ 说明：助手所在的是隔离云沙箱，**路由不到你的私网 IP，也没有你机器的凭据**，因此无法代你远程登录执行。
> 下面这套脚本是为"你在自己的 SSH 会话里执行"或"在那台机器上跑一个本地 agent"准备的。

---

## 路线 A（推荐）：在 GPU 机器上跑本地 agent

在算力机本地装一个 Claude Code（或同类本地 agent），它在机器本地就有 shell + GPU + 你之前的工作与模型，可以真正"新建目录 / 跑实验 / 验证评测"。把本目录(`ear_sql/`)放到机器上，让本地 agent 按本手册执行即可。这是让 agent 真正在你机器上闭环的唯一干净方式。

## 路线 B：你手动执行（脚本已备好）

### 0. 上传代码骨架
```bash
scp -r ear_sql dameng@192.168.1.141:~/
ssh dameng@192.168.1.141
```

### 1. 一键引导（建目录 + 环境 + 机器体检 + smoke test）
```bash
bash ~/ear_sql/scripts/remote_setup_and_run.sh
```
它会：
1. 新建带时间戳的运行目录 `~/ear_sql_runs/run_YYYYmmdd_HHMMSS/`（含 `logs/ ckpts/ preds/ data/ reports/`）。
2. 建 venv 并安装 `requirements.txt`。
3. 输出机器体检 `reports/machine.txt`（GPU、CUDA、torch）。
4. 扫描常见路径下你已有的模型/checkpoint → `reports/found_models.txt`。
5. 检查 BIRD 数据；缺失则在 `reports/data_TODO.txt` 写明摆放方式。
6. **跑 smoke test**：在真机上用临时 SQLite 验证 `执行/结果集比对/执行奖励/恢复奖励/动作重采样全流程`（13 项断言）。这一步**不需要 GPU、不需要模型**，用来确认管线在你机器上是通的。

预期 smoke test 结尾打印：`SMOKE TEST: ALL GREEN`。

### 2. 配置模型与数据
- 看 `reports/found_models.txt`，把你的基座 / SFT 初始化 checkpoint 路径填进 `ear_sql/configs/ear_sql.yaml` 的 `model.base` 和 `model.sft_init`。
- 按 `reports/data_TODO.txt` 摆好 BIRD：
  ```
  data/bird/dev.json
  data/bird/dev_databases/<db_id>/<db_id>.sqlite
  ```
  已有副本就软链：`ln -s /你的路径/bird  ~/ear_sql_runs/run_*/data/bird`

### 3. 接训练后端（必须，一次性工作）
代码骨架里 `src/grpo_trainer.py` 的 `Policy` 有三个 `# TODO`，需绑定到你的 RL 栈（**TRL / verl / OpenRLHF** 三选一）+ **vLLM** 做 rollout：
- `generate_group(prompt, g)`：采样 g 条 rollout，返回解码文本 + 动作 token 的 logprob。
- `generate_from_prefix(prompt, prefix, k)`：在"固定前缀"条件下采样 k 条后续（EAR-SQL 关键）。
- `ppo_update(advantages)`：PPO-clip 更新，应用 `compute_advantages` 给出的分离优势流（前缀 token 用 A_prefix，后续 token 用 A_suffix，掩掉前缀）。

> 建议先用 7B + 一张卡把流程跑通，再扩到 14B/32B。

### 4. 跑基线与 EAR-SQL
```bash
cd ~/ear_sql_runs/run_*/        # 进入本次运行目录
# 基线：SFT+GRPO（关闭重采样）
STAGE=baseline bash ~/ear_sql/scripts/remote_setup_and_run.sh
# EAR-SQL：开启动作重采样
STAGE=ear      bash ~/ear_sql/scripts/remote_setup_and_run.sh
```
日志在 `logs/baseline.log`、`logs/ear_sql.log`。训练循环每步会打印关键指标：
`all_wrong_groups`（全错组数）、`resampled_prefixes`、`recovered`（被救回的难题数）——
这几个就是验证 EAR-SQL 是否在起作用的直接证据。

### 5. 评测（EX / VES）
生成预测文件 `preds/preds.jsonl`（每行 `{"idx":int,"sql":str}`），然后：
```bash
PRED=preds/preds.jsonl STAGE=eval bash ~/ear_sql/scripts/remote_setup_and_run.sh
```
结果写入 `reports/eval_bird.txt`（EX 与 VES）。

### 6. 对照与结论
- **主对照**：EAR-SQL vs. 基线 GRPO，看 BIRD test EX 是否 +1~2pp，且增益是否集中在难度高的分桶。
- **算力对照**：再跑一版 `+2× rollout 预算` 的基线，EAR-SQL 应当胜过它（证明收益不是单纯多花算力）。
- **严谨性**：在 FLEX / 人工核验子集上复评，避免被标注噪声误导。

---

## 一次完整跑通的顺序（速查）
```bash
scp -r ear_sql dameng@192.168.1.141:~/ && ssh dameng@192.168.1.141
bash ~/ear_sql/scripts/remote_setup_and_run.sh          # 环境 + smoke(全绿)
# 填 configs/ear_sql.yaml(模型) + 摆好 BIRD 数据 + 接好 Policy 三个 TODO
cd ~/ear_sql_runs/run_*/
STAGE=baseline bash ~/ear_sql/scripts/remote_setup_and_run.sh
STAGE=ear      bash ~/ear_sql/scripts/remote_setup_and_run.sh
PRED=preds/preds.jsonl STAGE=eval bash ~/ear_sql/scripts/remote_setup_and_run.sh
```

## 我能继续帮的
- 把 `Policy` 的 TODO 按你选定的栈（TRL / verl）写成具体实现。
- 加一个"难度分桶 + 全错组转化率"的统计脚本，自动产出对照表。
- 如果你能给我一条可达的执行通道（如机器上的本地 agent、或一个远程执行 connector），我可以直接驱动整个流程。
