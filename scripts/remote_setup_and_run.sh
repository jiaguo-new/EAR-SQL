#!/usr/bin/env bash
# =============================================================================
# EAR-SQL remote bootstrap — run THIS on the compute machine (dameng@192.168.1.141)
#
#   scp -r ear_sql dameng@192.168.1.141:~/            # 1. upload the skeleton
#   ssh dameng@192.168.1.141                          # 2. log in
#   bash ~/ear_sql/scripts/remote_setup_and_run.sh    # 3. run this
#
# It creates a fresh dated run dir, sets up the env, verifies GPU, fetches data,
# runs a VERIFIABLE smoke test of the implemented components, and (when you set
# STAGE=train/eval) launches the baseline / EAR-SQL / evaluation stages.
#
# Stages (set with STAGE=...):
#   env     : create venv + install deps                      (default also runs)
#   smoke   : unit-test reward/exec/resampling on real data   (default)
#   baseline: SFT+GRPO reproduction (needs your RL stack)      (opt-in)
#   ear     : EAR-SQL run                                      (opt-in)
#   eval    : BIRD EX/VES on a predictions file               (opt-in)
# =============================================================================
set -euo pipefail

STAGE="${STAGE:-smoke}"
EAR_SRC="$(cd "$(dirname "$0")/.." && pwd)"          # the uploaded ear_sql/ dir
RUN_ROOT="${RUN_ROOT:-$HOME/ear_sql_runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/run_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-python3}"

log(){ printf '\033[1;36m[EAR-SQL]\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31m[EAR-SQL][FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

log "source skeleton : $EAR_SRC"
log "run directory   : $RUN_DIR"
mkdir -p "$RUN_DIR"/{logs,ckpts,preds,data,reports}
cd "$RUN_DIR"

# ---- 1. environment --------------------------------------------------------
log "== STAGE env: python venv + deps =="
if [ ! -d "$RUN_DIR/.venv" ]; then
  "$PY" -m venv "$RUN_DIR/.venv"
fi
# shellcheck disable=SC1091
source "$RUN_DIR/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$EAR_SRC/requirements.txt" || log "WARN: some deps failed; smoke test only needs sqlglot+func-timeout+numpy"

# ---- 2. machine / GPU report ----------------------------------------------
log "== machine report =="
{
  echo "host: $(hostname)"; echo "date: $(date -Is)"
  echo "python: $($PY --version 2>&1)"
  echo "--- nvidia-smi ---"; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>/dev/null || echo "no nvidia-smi"
  echo "--- torch cuda ---"; python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())" 2>/dev/null || echo "torch not importable"
} | tee "$RUN_DIR/reports/machine.txt"

# ---- 3. discover prior work (models / checkpoints) ------------------------
log "== scanning for existing models/checkpoints (your prior work) =="
{
  echo "# edit these globs to match your machine"
  for d in "$HOME/models" "$HOME/ckpts" "$HOME/checkpoints" "/data/models" "/mnt/models" "$HOME/.cache/huggingface/hub"; do
    if [ -d "$d" ]; then
      echo "## $d"
      ls -1 "$d" 2>/dev/null | head -20 || true
    fi
  done
} | tee "$RUN_DIR/reports/found_models.txt"
log "review reports/found_models.txt and set BASE_MODEL / SFT_INIT in configs/ear_sql.yaml accordingly"

# ---- 4. data ---------------------------------------------------------------
log "== checking BIRD data =="
BIRD_DIR="${BIRD_DIR:-$RUN_DIR/data/bird}"
if [ ! -f "$BIRD_DIR/dev.json" ]; then
  cat <<EOF | tee "$RUN_DIR/reports/data_TODO.txt"
BIRD dev not found at $BIRD_DIR/dev.json
Place (or symlink) the BIRD dev set so the layout is:
  $BIRD_DIR/dev.json
  $BIRD_DIR/dev_databases/<db_id>/<db_id>.sqlite
Official data: https://bird-bench.github.io/  (or reuse your existing copy:
  ln -s /path/to/your/bird $BIRD_DIR )
EOF
else
  log "found BIRD dev: $BIRD_DIR/dev.json"
fi

# ---- 5. SMOKE TEST (always runnable, proves the implemented parts work) ----
if [ "$STAGE" = "smoke" ] || [ "$STAGE" = "all" ]; then
  log "== STAGE smoke: unit-test reward / sql_exec / resampling on a real sqlite =="
  export PYTHONPATH="$EAR_SRC"
  python "$EAR_SRC/scripts/smoke_test.py" 2>&1 | tee "$RUN_DIR/logs/smoke.log"
  log "smoke test done -> logs/smoke.log"
fi

# ---- 6. BASELINE (opt-in; needs your RL stack wired into grpo_trainer.py) --
if [ "$STAGE" = "baseline" ] || [ "$STAGE" = "all" ]; then
  log "== STAGE baseline: SFT+GRPO reproduction =="
  log "Prereq: implement Policy.generate_group / ppo_update in src/grpo_trainer.py"
  log "        (bind to TRL/verl/OpenRLHF + a vLLM rollout server). Then:"
  RESAMPLE_ENABLED=false python -m src.grpo_trainer "$EAR_SRC/configs/ear_sql.yaml" \
      2>&1 | tee "$RUN_DIR/logs/baseline.log" || die "baseline failed (most likely Policy not implemented yet)"
fi

# ---- 7. EAR-SQL run (opt-in) ----------------------------------------------
if [ "$STAGE" = "ear" ] || [ "$STAGE" = "all" ]; then
  log "== STAGE ear: EAR-SQL (resample.enabled=true) =="
  python -m src.grpo_trainer "$EAR_SRC/configs/ear_sql.yaml" \
      2>&1 | tee "$RUN_DIR/logs/ear_sql.log" || die "ear run failed (check Policy + GPU mem)"
fi

# ---- 8. EVAL (opt-in) ------------------------------------------------------
if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
  log "== STAGE eval: BIRD EX/VES =="
  PRED="${PRED:-$RUN_DIR/preds/preds.jsonl}"
  [ -f "$PRED" ] || die "no predictions at $PRED (produce preds.jsonl: {\"idx\":int,\"sql\":str} per line)"
  python "$EAR_SRC/eval/eval_bird.py" --pred "$PRED" \
      --gold "$BIRD_DIR/dev.json" --db_root "$BIRD_DIR/dev_databases" \
      2>&1 | tee "$RUN_DIR/reports/eval_bird.txt"
fi

log "ALL DONE. Artifacts under: $RUN_DIR"
log "  reports/machine.txt   reports/found_models.txt   logs/   reports/eval_bird.txt"
