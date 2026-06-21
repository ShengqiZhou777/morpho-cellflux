#!/usr/bin/env bash
# Auto-launch the v9 idsig run AFTER run_new_panels.sh finishes diet-v3 -> v8.
# Keyed on the launcher's own "ALL DONE" marker in run_new_panels.log (written only after v8
# exits), so there is no process race and no risk of firing in the diet-v3 -> v8 gap.
# Detached via setsid; survives the Claude session. Progress -> outputs/idsig_watcher.log,
# training -> outputs/run_idsig.log (and outputs/cellflux_pm_train_id_v9/train_stdout.log).
set -uo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
LOG="$PROJECT/outputs/idsig_watcher.log"
echo "[$(date -Is)] idsig watcher started (pid $$); waiting for 'ALL DONE' (diet-v3 + v8 finished)" >> "$LOG"
while ! grep -q "ALL DONE" "$PROJECT/outputs/run_new_panels.log" 2>/dev/null; do
  sleep 120
done
echo "[$(date -Is)] launcher ALL DONE detected; settling 30s then launching v9 (idsig)" >> "$LOG"
sleep 30
bash scripts/run_idsig.sh >> "$PROJECT/outputs/run_idsig.log" 2>&1
echo "[$(date -Is)] v9 (idsig) exited code $?" >> "$LOG"
