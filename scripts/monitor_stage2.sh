#!/usr/bin/env bash
# Monitor Stage 2 (62D omics) training progress
cd "$(dirname "$0")/.."

RUN=$(ls -td outputs/runs/microalgae/timepoint_512_62d_e40_* 2>/dev/null | head -1)
echo "=== Stage 2 训练监控 ==="
echo "Run: $RUN"
echo ""

# 进程状态
if ps aux | grep -q "[p]henoflux.train.*62d\|[p]henoflux.train --dataset"; then
  echo "✓ 训练进程存活"
else
  echo "✗ 训练进程未找到（可能已完成或崩溃）"
fi

# GPU
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo ""

# checkpoints
echo "已保存 checkpoints:"
ls -t "$RUN"/checkpoint-*.pth 2>/dev/null | head -5 | xargs -I{} basename {} 2>/dev/null || echo "  (尚无)"
echo ""

# epoch 汇总 + FID
echo "Epoch 汇总 (loss + FID):"
cat "$RUN/log.txt" 2>/dev/null | tail -8 || echo "  (尚无 epoch 完成)"
