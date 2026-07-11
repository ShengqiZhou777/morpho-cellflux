#!/usr/bin/env bash
# Stage 2 前台启动：屏幕只显示 tqdm 的 epoch 进度条 (X/40)
# 每50步的 loss 日志(stdout)写到 <output_dir>/train_steps.log，不刷屏
#
# 用法（自己在终端里跑，会阻塞占用这个终端）:
#   bash scripts/run_stage2_foreground.sh
#
# 屏幕显示（tqdm，原地刷新一行）:
#   12%|████▊              | 5/40 [07:03:12<49:21:00, 5075s/it]
#
# 另开终端看每步loss/FID:  tail -f <output_dir>/train_steps.log
# 看每epoch汇总(JSON):     tail -f <output_dir>/log.txt

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="outputs/runs/microalgae/timepoint_512_62d_e40_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=== Stage 2: 62D 组学条件训练（前台）==="
echo "输出目录: $OUTPUT_DIR"
echo "屏幕 = tqdm epoch 进度条 | 每步 loss = $OUTPUT_DIR/train_steps.log"
echo "Ctrl+C 中断。开始..."
echo ""

# 分流的关键（前台阻塞，无后台管道，无缓冲问题）：
#   1> train_steps.log : stdout(logger 每50步 loss, train.py:55) 进文件
#   2>&1 到终端        : stderr(tqdm 进度条) 保留在终端，\r 原地刷新正常工作
torchrun --standalone --nproc_per_node=1 -m phenoflux.train \
  --dataset phenoflux \
  --config microalgae_timepoint_512_62d \
  --batch_size 12 \
  --epochs 40 \
  --use_initial 1 \
  --cfg_scale 0.2 \
  --use_ema \
  --skewed_timesteps \
  --class_drop_prob 0.2 \
  --eval_frequency 5 \
  --fid_samples 512 \
  --compute_fid \
  --output_dir "$OUTPUT_DIR" \
  1> "$OUTPUT_DIR/train_steps.log"

echo ""
echo "✓ 训练结束: $OUTPUT_DIR"
