#!/usr/bin/env bash
# 监控训练进度的便捷脚本

cd "$(dirname "$0")/.."

echo "=== PhenoFlux 训练监控 ==="
echo ""

# 找到最新的 convergence run
OUT_DIR=$(ls -td outputs/runs/microalgae/timepoint_512_convergence_e40_* 2>/dev/null | head -1)
if [ -z "$OUT_DIR" ]; then
  echo "ERROR: 未找到 convergence run"
  exit 1
fi

echo "训练目录: $OUT_DIR"
echo ""

# 找到后台任务输出文件
TASK_OUTPUT=$(find /tmp/claude-1000 -name "*.output" -newer "$OUT_DIR/args.json" 2>/dev/null | head -1)

if [ -z "$TASK_OUTPUT" ]; then
  echo "警告: 未找到后台任务输出文件，显示 log.txt（如果存在）"
  if [ -f "$OUT_DIR/log.txt" ]; then
    tail -f "$OUT_DIR/log.txt"
  else
    echo "log.txt 尚未生成。训练日志将在 epoch 结束时写入。"
    echo ""
    echo "GPU 状态:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  fi
else
  echo "实时日志: $TASK_OUTPUT"
  echo ""
  echo "=== 最新 20 行 ==="
  tail -20 "$TASK_OUTPUT"
  echo ""
  echo "--- 实时追踪模式（Ctrl+C 退出）---"
  tail -f "$TASK_OUTPUT"
fi
