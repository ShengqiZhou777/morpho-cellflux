## 训练监控指南

**训练状态**: ✅ 运行中（Epoch 5/40，步数 1850/4298）

### 快速监控命令

```bash
# 实时训练日志
bash scripts/monitor_training.sh

# GPU 使用率
watch -n 5 nvidia-smi

# 进程检查
ps aux | grep phenoflux.train | grep -v grep

# 查看当前 epoch 进度（从后台任务输出）
tail -5 /tmp/claude-1000/-home-shockley-myproject-PhenoFlux-morpho-cellflux/*/tasks/bw535x40t.output
```

### 日志说明

**日志位置两处**：
1. **实时日志**（训练过程中）：
   - `/tmp/claude-1000/.../tasks/bw535x40t.output`
   - 每 50 步打印一次 loss
   - 用 `bash scripts/monitor_training.sh` 实时追踪

2. **汇总日志**（epoch 结束后）：
   - `outputs/runs/microalgae/timepoint_512_convergence_e40_*/log.txt`
   - 每 epoch 一行 JSON（格式：`{"train_loss": ..., "epoch": ...}`）
   - Epoch 5 结束后会出现第一行

### 预计时间线

| Epoch | 预计完成时间 | 说明 |
|-------|------------|------|
| 5 | 2026-07-07 23:40 | 当前（Resume 重复跑，验证恢复正确） |
| 10 | 2026-07-08 05:20 | 第一个 FID 评估点 |
| 20 | 2026-07-08 16:40 | 中期检查点 |
| 30 | 2026-07-09 04:00 | 后期检查点 |
| 40 | 2026-07-09 15:20 | 完成 |

**单 epoch 约需 1.4 小时**（4298 步 × 18 秒/步），35 epochs 约 **49 小时**。

### 检查点和报告

**Checkpoints**（每 epoch 保存）：
```
outputs/runs/.../checkpoint-{5,10,15,20,25,30,35,40}.pth
outputs/runs/.../checkpoint-best_loss.pth  # 最低 loss
outputs/runs/.../checkpoint.pth            # 最新
```

**Visual reports**（训练完成后生成）：
```bash
# 生成各 epoch 的 control/generated/target 对比图
bash scripts/generate_convergence_reports.sh \
  outputs/runs/microalgae/timepoint_512_convergence_e40_20260707_221415
```

### 如何判断成功

训练完成后，运行：
```python
import numpy as np, glob, os
from PIL import Image

reports = sorted(glob.glob('outputs/reports/convergence_*_ep*'))
for r in reports:
    ep = r.split('_ep')[1].split('_')[0]
    diffs = []
    for g in glob.glob(f'{r}/*/*_generated.png')[:12]:
        c = g.replace('_generated','_control')
        if not os.path.exists(c): continue
        diffs.append(np.abs(
            np.array(Image.open(g))[:,:,:3].astype(float) -
            np.array(Image.open(c))[:,:,:3].astype(float)
        ).mean())
    if diffs:
        mean_diff = np.mean(diffs)
        print(f'Epoch {ep:>2}: 像素差 = {mean_diff:.1f}')
        if mean_diff > 5:
            print(f'  ✓ 成功！模型开始学习 control→target 变换')
        elif mean_diff < 3:
            print(f'  ✗ 仍为恒等映射，需进入阶段 2（组学扩充）')
```

**期望**：epoch 40 时像素差 **> 5**，说明欠训练是主因，任务完成。
