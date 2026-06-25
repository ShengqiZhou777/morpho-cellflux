#!/usr/bin/env bash
# Auto-eval: waits for training to reach epoch 5, then runs gap_closed + MoA.
# Run this alongside training:
#   nohup bash scripts/auto_eval_epoch5.sh > outputs/paper/info_control/auto_eval.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pmf

RUN_DIR="outputs/paper/info_control"
EPOCH=5

echo "[$(date)] Waiting for epoch $EPOCH eval to complete..."
echo "[$(date)] Watching: $RUN_DIR/fid_samples/epoch-$EPOCH/"

# Wait for the epoch dir to appear and generation to finish
while [ ! -d "$RUN_DIR/fid_samples/epoch-$EPOCH" ]; do
  sleep 60
done

# Wait a bit more in case FID computation is still running
sleep 30

echo "[$(date)] Epoch $EPOCH found! Running gap_closed..."

# Step 1: gap_closed
python scripts/diet_marker_distribution_figure.py \
  --run-dir "$RUN_DIR" \
  --epoch "$EPOCH" \
  --out-dir "$RUN_DIR" \
  --prefix "info_control_ep${EPOCH}" \
  2>&1 | tee "$RUN_DIR/gap_closed_ep${EPOCH}.log"

echo "[$(date)] gap_closed done. Running MoA..."

# Step 2: MoA
python src/morphoflux/engine/moa/train_moa.py \
  --config_path configs/diet_id_18ch.yaml \
  --mode eval \
  --img_root_path "$RUN_DIR/fid_samples/epoch-$EPOCH" \
  --ckpt_path outputs/baselines/moa/diet/condition_classifier.pth \
  --out_json "$RUN_DIR/moa_ep${EPOCH}.json" \
  2>&1 | tee "$RUN_DIR/moa_ep${EPOCH}.log"

echo "[$(date)] All done!"

# Print summary
echo ""
echo "============================================"
echo "  Epoch $EPOCH Results"
echo "============================================"
python3 -c "
import json
gap_f = 'outputs/paper/info_control/info_control_ep${EPOCH}_marker_distribution_summary.json'
moa_f = 'outputs/paper/info_control/moa_ep${EPOCH}.json'

print('--- gap_closed ---')
g = json.load(open(gap_f))
for s in g['summary']:
    print(f'  {s[\"condition\"]:6s} {s[\"marker\"]:15s} gap_closed={s[\"gap_closed\"]:+.4f}  gen_mean={s[\"generated_mean\"]:.4f}  tgt_mean={s[\"target_mean\"]:.4f}')

print()
print('--- MoA ---')
m = json.load(open(moa_f))
print(f'  Accuracy: {m[\"moa_acc\"]:.2f}%')
if 'per_class' in m:
    for cls, v in m['per_class'].items():
        print(f'  {cls}: {v[\"acc\"]:.2f}% (n={v[\"n\"]})')
"
echo "[$(date)] Auto-eval complete."