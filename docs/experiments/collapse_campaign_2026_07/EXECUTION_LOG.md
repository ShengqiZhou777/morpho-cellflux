## 恒等映射问题 - 执行总结

**日期**: 2026-07-07 22:15  
**状态**: 阶段 1 执行中

---

### ✅ 已完成

**1. 根因诊断**
- 模型学成恒等映射（generated ≈ control，像素差 <2）
- 三重根因：随机配对 + 严重欠训练（5 epoch）+ 弱条件（4维）
- 详见 `ACTION_PLAN.md`

**2. 训练启动**
- **运行中**: 40-epoch 收敛测试
- 起点: epoch 5 checkpoint（`timepoint_512_b12_e5_20260706_000411/checkpoint.pth`）
- 目标: epoch 40
- 硬件: 1× RTX 4090 (24GB), GPU 利用率 100%
- 输出: `outputs/runs/microalgae/timepoint_512_convergence_e40_20260707_221415/`
- **预计完成**: 2026-07-08 16:15 (约 18 小时)

**3. 清理完成**
- 删除 init0/init2 发散分支（3 个 runs + 4 个 reports）
- 释放磁盘空间: **13.2 GB**
- 保留: init1 baseline + 当前 convergence run

**4. 脚本/文档**
- 新增:
  - `ACTION_PLAN.md` — 完整诊断和 3 阶段收束方案
  - `scripts/train_convergence_test.sh` — 40-epoch 训练
  - `scripts/generate_convergence_reports.sh` — 多 epoch visual reports
  - `scripts/generate_report.sh` — 标准化报告生成
  - `EXECUTION_LOG.md` — 本文件
- 修正:
  - `CLAUDE.md` GPU 信息（2×5090 → 1×4090）
  - `CLAUDE.md` base_condition_dim（61 → 4, 标注待扩展）

---

### 📊 监控命令

```bash
# 实时训练日志（JSON 格式，每 epoch 一行）
tail -f outputs/runs/microalgae/timepoint_512_convergence_e40_20260707_221415/log.txt

# GPU 使用率（每 5 秒刷新）
watch -n 5 nvidia-smi

# 检查进程存活
ps aux | grep phenoflux.train | grep -v grep

# 当前 epoch 和 loss（需要 log.txt 生成后）
tail -1 outputs/runs/microalgae/timepoint_512_convergence_e40_20260707_221415/log.txt | jq '.epoch, .train_loss'
```

---

### 🔍 下一步（训练完成后）

**A. 生成 visual reports（~30 min）**
```bash
bash scripts/generate_convergence_reports.sh \
  outputs/runs/microalgae/timepoint_512_convergence_e40_20260707_221415
```

**B. 量化对比（立即可用）**
```python
# 对比各 epoch 的 control vs generated 像素差
import numpy as np, glob
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
        print(f'Epoch {ep:>2}: 像素差 = {np.mean(diffs):.1f} ± {np.std(diffs):.1f}')
```

**C. 判断标准**
- **成功**（欠训练是主因）：epoch 40 时像素差 **> 5**，FID < 80
  - → 直接用满训练的模型，任务完成
- **失败**（配对/条件是主因）：像素差仍 **< 3**
  - → 进入阶段 2：组学条件扩充到 61 维
  - → 或阶段 3：重审任务定义（接受群体均值 vs 改配对策略）

---

### 📌 关键发现记录

1. **GPU 信息错误**: 文档说 2×5090，实际 1×4090 → 训练时间翻倍
2. **配对机制**: `batch_random` = 同光照条件内随机配对，**不是**同细胞时序对
3. **组学数据可用**: 9 整数小时点，可插值到 105 个 5 分钟 bin
4. **checkpoint 结构**: 包含 epoch/model/optimizer/lr_schedule，支持完整 resume

---

**等待**: 训练完成（2026-07-08 16:15 预计）  
**后续**: 根据 epoch 40 结果决定是否需要阶段 2
