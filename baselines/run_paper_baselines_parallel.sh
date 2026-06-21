#!/usr/bin/env bash
# Parallel paper-baseline queue: one benchmark per GPU, run concurrently.
#
# Rationale: each training job only uses ~7-14 GiB of a 32 GiB card, so the
# sequential queue (run_paper_baselines.sh) left a whole GPU idle. This variant
# pins each benchmark to one GPU and runs the two benchmarks side by side.
#
# Training recipes are preserved, NOT changed:
#   - PhenDiff runs single-process (NPROC=1) at batch 32, which is the SAME
#     global batch as the 2-GPU NPROC=2 batch-16 setup.
#   - IMPA's global batch is unchanged (DataParallel over 1 visible GPU == the
#     same nominal batch, just on one card).
#
# Resumable: any method/benchmark with aggregate_eval_summary.json is skipped.
set -uo pipefail   # deliberately no -e: one lane must not silently kill the other

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

INCLUDE_STARGAN="${INCLUDE_STARGAN:-0}"   # StarGAN stays a separate supplement
PHENDIFF_EPOCHS="${PHENDIFF_EPOCHS:-8}"
IMPA_EPOCHS="${IMPA_EPOCHS:-8}"
PHENDIFF_BATCH="${PHENDIFF_BATCH:-32}"    # single-GPU == 2-GPU(batch16) global batch
IMPA_BATCH="${IMPA_BATCH:-16}"
# Lane assignment: "<gpu>:<benchmark>" pairs.
LANE0="${LANE0:-0:diet}"
LANE1="${LANE1:-1:crispr}"
RUN_ID="${BASELINE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
STATUS_ROOT="$PROJECT/outputs/baselines/logs/status"
STATUS_DIR="$STATUS_ROOT/$RUN_ID"
STATUS_JSON="$PROJECT/outputs/baselines/logs/paper_baselines_parallel_status.json"

# Concurrency hardening (two PhenDiff jobs at once previously raced on startup
# network calls and a shared accelerate rendezvous port):
#   - run fully offline; all weights (incl. torch-fidelity InceptionV3) are cached
#   - stagger lane1 so lane0 warms shared caches / passes startup first
#   - each lane gets its own accelerate main_process_port (set per-lane below)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export GIT_TERMINAL_PROMPT=0
LANE_STAGGER_SECONDS="${LANE_STAGGER_SECONDS:-120}"

mkdir -p "$STATUS_DIR"

summary_exists() { [[ -f "$PROJECT/$1/aggregate_eval_summary.json" ]]; }

record_status() {
  local benchmark="$1" method="$2" status="$3" message="${4:-}" code="${5:-0}" gpu="${6:-}"
  local lane_key="${gpu:-cpu}"
  python - "$STATUS_DIR/${benchmark}_${method}_${lane_key}.json" "$benchmark" "$method" "$status" "$message" "$code" "$gpu" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, benchmark, method, status, message, code, gpu = sys.argv[1:8]
Path(out).write_text(json.dumps({
    "time": datetime.now(timezone.utc).isoformat(),
    "benchmark": benchmark,
    "method": method,
    "status": status,
    "message": message,
    "exit_code": int(code),
    "gpu": gpu or None,
}, indent=2) + "\n")
PY
}

write_status_json() {
  local final_rc="$1" lane0_rc="${2:-}" lane1_rc="${3:-}" collect_rc="${4:-}"
  python - "$STATUS_JSON" "$STATUS_DIR" "$RUN_ID" "$final_rc" "$lane0_rc" "$lane1_rc" "$collect_rc" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, status_dir, run_id, final_rc, lane0_rc, lane1_rc, collect_rc = sys.argv[1:8]
status_path = Path(status_dir)
records = []
if status_path.exists():
    for path in sorted(status_path.glob("*.json")):
        records.append(json.loads(path.read_text()))
failed = [r for r in records if r.get("status") == "failed"]
payload = {
    "run_id": run_id,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "ok": int(final_rc) == 0 and not failed,
    "final_exit_code": int(final_rc),
    "lane_exit_codes": {
        "lane0": None if lane0_rc == "" else int(lane0_rc),
        "lane1": None if lane1_rc == "" else int(lane1_rc),
    },
    "collect_exit_code": None if collect_rc == "" else int(collect_rc),
    "status_dir": str(status_path),
    "records": records,
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {out}")
PY
}

benchmark_config() {
  case "$1" in
    diet) echo "configs/diet_id.yaml" ;;
    crispr) echo "configs/perturbmulti_train_id.yaml" ;;
    *) echo "Unknown benchmark: $1" >&2; return 2 ;;
  esac
}

ensure_export() {
  local benchmark="$1" config out
  config="$(benchmark_config "$benchmark")"
  out="outputs/baselines/_data/$benchmark"
  if [[ -f "$out/manifest.json" ]]; then
    echo "[$(date -Is)] SKIP export $benchmark: manifest exists"
    return 0
  fi
  echo "[$(date -Is)] START export $benchmark"
  python baselines/export_baseline_data.py \
    --config "$config" --benchmark "$benchmark" --output "$out" \
    --splits train,test --workers "${EXPORT_WORKERS:-8}"
  echo "[$(date -Is)] DONE export $benchmark"
}

run_copy_control_for() {
  local benchmark="$1" config out
  config="$(benchmark_config "$benchmark")"
  out="outputs/baselines/copy_control/$benchmark"
  if summary_exists "$out"; then
    echo "[$(date -Is)] SKIP copy_control $benchmark: summary exists"
    return 0
  fi
  echo "[$(date -Is)] START copy_control $benchmark"
  python baselines/copy_control.py --config "$config" --output "$out" --split test \
    && python scripts/aggregate_eval.py "$out" 5 0
  echo "[$(date -Is)] DONE copy_control $benchmark"
}

# A lane owns one GPU and runs the full per-benchmark chain on it, sequentially.
lane() {
  local spec="$1" gpu benchmark
  gpu="${spec%%:*}"
  benchmark="${spec#*:}"
  local lane_rc=0 code=0
  export CUDA_VISIBLE_DEVICES="$gpu"   # subshell-local: pins every step in this lane
  export MAIN_PROCESS_PORT=$((29500 + gpu))   # distinct accelerate rendezvous per lane

  echo "[$(date -Is)] [gpu=$gpu] LANE START $benchmark"

  if summary_exists "outputs/baselines/copy_control/$benchmark"; then
    echo "[$(date -Is)] [gpu=$gpu] SKIP copy_control $benchmark: summary exists"
    record_status "$benchmark" copy_control skipped "summary exists" 0 "$gpu"
  elif run_copy_control_for "$benchmark"; then
    record_status "$benchmark" copy_control complete "completed" 0 "$gpu"
  else
    code=$?
    echo "[$(date -Is)] [gpu=$gpu] FAIL copy_control $benchmark exit=$code"
    record_status "$benchmark" copy_control failed "copy-control or aggregate_eval failed" "$code" "$gpu"
    lane_rc=1
  fi

  if summary_exists "outputs/baselines/phendiff/$benchmark"; then
    echo "[$(date -Is)] [gpu=$gpu] SKIP PhenDiff $benchmark: summary exists"
    record_status "$benchmark" phendiff skipped "summary exists" 0 "$gpu"
  else
    echo "[$(date -Is)] [gpu=$gpu] START PhenDiff $benchmark (NPROC=1 BATCH=$PHENDIFF_BATCH)"
    if NPROC=1 BENCHMARK="$benchmark" EPOCHS="$PHENDIFF_EPOCHS" BATCH="$PHENDIFF_BATCH" \
      bash baselines/run_phendiff.sh; then
      record_status "$benchmark" phendiff complete "completed" 0 "$gpu"
    else
      code=$?
      echo "[$(date -Is)] [gpu=$gpu] FAIL PhenDiff $benchmark exit=$code"
      record_status "$benchmark" phendiff failed "run_phendiff.sh failed" "$code" "$gpu"
      lane_rc=1
    fi
  fi

  if summary_exists "outputs/baselines/impa/$benchmark"; then
    echo "[$(date -Is)] [gpu=$gpu] SKIP IMPA $benchmark: summary exists"
    record_status "$benchmark" impa skipped "summary exists" 0 "$gpu"
  else
    echo "[$(date -Is)] [gpu=$gpu] START IMPA $benchmark (BATCH=$IMPA_BATCH)"
    if BENCHMARK="$benchmark" EPOCHS="$IMPA_EPOCHS" BATCH="$IMPA_BATCH" DEVICES=1 \
      bash baselines/run_impa.sh; then
      record_status "$benchmark" impa complete "completed" 0 "$gpu"
    else
      code=$?
      echo "[$(date -Is)] [gpu=$gpu] FAIL IMPA $benchmark exit=$code"
      record_status "$benchmark" impa failed "run_impa.sh failed" "$code" "$gpu"
      lane_rc=1
    fi
  fi

  echo "[$(date -Is)] [gpu=$gpu] LANE DONE $benchmark exit=$lane_rc"
  return "$lane_rc"
}

echo "[$(date -Is)] START parallel paper baseline queue"
echo "project=$PROJECT"
echo "run_id=$RUN_ID"
echo "lane0=$LANE0 lane1=$LANE1 include_stargan=$INCLUDE_STARGAN"
echo "phendiff_epochs=$PHENDIFF_EPOCHS phendiff_batch=$PHENDIFF_BATCH impa_epochs=$IMPA_EPOCHS impa_batch=$IMPA_BATCH"

# Exports are CPU/IO; do them up front so both GPU lanes start clean.
export_rc=0
for spec in "$LANE0" "$LANE1"; do
  benchmark="${spec#*:}"
  if [[ -f "outputs/baselines/_data/$benchmark/manifest.json" ]]; then
    record_status "$benchmark" export skipped "manifest exists" 0 ""
    echo "[$(date -Is)] SKIP export $benchmark: manifest exists"
  elif ensure_export "$benchmark"; then
    record_status "$benchmark" export complete "completed" 0 ""
  else
    code=$?
    record_status "$benchmark" export failed "export failed" "$code" ""
    export_rc=1
  fi
done

if [[ "$export_rc" -ne 0 ]]; then
  echo "[$(date -Is)] export failed; not starting GPU lanes"
  write_status_json 1 "" "" ""
  exit 1
fi

lane "$LANE0" &
p0=$!
sleep "$LANE_STAGGER_SECONDS"   # let lane0 pass startup before lane1 begins
lane "$LANE1" &
p1=$!
wait "$p0"; r0=$?
wait "$p1"; r1=$?
echo "[$(date -Is)] lanes finished: lane0=$r0 lane1=$r1"

collect_rc=0
python baselines/collect_paper_metrics.py || collect_rc=$?

final_rc=0
if [[ "$r0" -ne 0 || "$r1" -ne 0 || "$collect_rc" -ne 0 ]]; then
  final_rc=1
fi
write_status_json "$final_rc" "$r0" "$r1" "$collect_rc"
echo "[$(date -Is)] DONE parallel paper baseline queue exit=$final_rc"
exit "$final_rc"
