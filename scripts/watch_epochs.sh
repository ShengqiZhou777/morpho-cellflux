#!/usr/bin/env bash
# 只显示 epoch 级进度：每个 epoch 完成时打印 loss + FID
# 用法: bash scripts/watch_epochs.sh
cd "$(dirname "$0")/.."

RUN=$(ls -td outputs/runs/microalgae/timepoint_512_62d_e40_* 2>/dev/null | head -1)
LOG="$RUN/log.txt"
TOTAL=40

echo "=== Epoch 级进度监控 ==="
echo "Run: $(basename "$RUN")"
echo "总 epoch: $TOTAL | 单 epoch ~1.4h"
echo "----------------------------------------"

# 先打印已完成的 epoch
if [ -f "$LOG" ]; then
  python3 - "$LOG" "$TOTAL" <<'PY'
import json, sys
log, total = sys.argv[1], int(sys.argv[2])
try:
    for line in open(log):
        d = json.loads(line)
        ep = d.get("epoch", "?")
        loss = d.get("train_loss", float("nan"))
        fid = d.get("eval_fid", d.get("fid", None))
        bar_n = int((ep+1)/total*30) if isinstance(ep,int) else 0
        bar = "█"*bar_n + "░"*(30-bar_n)
        fid_s = f" | FID={fid:.2f}" if fid is not None else ""
        print(f"Epoch {ep:>2}/{total} [{bar}] loss={loss:.5f}{fid_s}")
except Exception as e:
    print(f"(解析中: {e})")
PY
fi

echo "----------------------------------------"
echo "等待新 epoch 完成（Ctrl+C 退出）..."
# 实时追踪：每有新行写入 log.txt 就格式化打印
tail -f "$LOG" 2>/dev/null | while read -r line; do
  echo "$line" | python3 -c "
import json,sys
for l in sys.stdin:
    try:
        d=json.loads(l)
        ep=d.get('epoch','?'); loss=d.get('train_loss',float('nan'))
        fid=d.get('eval_fid',d.get('fid',None))
        n=int((ep+1)/$TOTAL*30) if isinstance(ep,int) else 0
        bar='█'*n+'░'*(30-n)
        fs=f' | FID={fid:.2f}' if fid is not None else ''
        print(f'Epoch {ep:>2}/$TOTAL [{bar}] loss={loss:.5f}{fs}')
    except: pass
"
done
