# PhenoFlux 恒等映射问题 - 收束行动方案

**诊断日期**: 2026-07-07  
**问题**: init1 模型学成恒等映射（generated ≈ control，像素差 <2）

---

## 根因分析

### 已确认事实

1. **配对机制**
   - `batch_random`: 每个 treated 从**同光照条件**的 control 池随机抽一个
   - control 和 target **不是同一细胞的时序对**
   - 每 epoch 重新洗牌，配对关系不固定

2. **数据特征**
   - Control: 0h 细胞（source_actual_time_h ≈ 0-0.4h）
   - Target: 同条件下 T 小时细胞（0.98-74h）
   - 同时间点内 control vs treated 像素差异：23-31（resize 到 64×64 后）

3. **训练状态**
   - 仅 **5 epoch**，train_loss 0.0127→0.0049 仍在陡降
   - **无 FID 评估**（FID 每 10 epoch 才计算）
   - 4 个平行实验分支（init0/1/2, noise0.2/0.5）发散中

4. **条件维度**
   - 当前：4 维（`cond_light, cond_dark, time_norm, time_bin_h`）
   - CLAUDE.md 声称 61 维 → **文档与实现脱节**
   - 组学数据可用：9 整数小时点，rna_pca_0~28 + prot_pca_0~28

### 根因推测（按可能性排序）

**① 严重欠训练**（最可能，最易验证）
- 5 epoch 远未收敛，模型可能还没开始学习非零速度场
- **验证方法**：跑满 40 epoch，看 generated 是否开始偏离 control

**② 随机配对 + 群体均值难题**（次要，但如果①不解决这个就是主因）
- 随机配对下，模型只能学"群体平均漂移"
- 如果 0h→Th 的群体形态变化小 → 最优解 ≈ 恒等映射
- **解决方向**：不改配对（接受群体均值任务），通过更强条件引导变化

**③ 条件太弱**（可能加剧①②）
- 4 维实际只有时间 1 个有效自由度
- **解决方向**：加入组学 PCA（需插值到 5 分钟粒度）

---

## 收束方案（3 阶段）

### 阶段 1：验证欠训练假设（优先级：P0，1 天内）

**目标**：用现有 config 跑满 40 epoch，确认恒等映射是否因欠训练而非配对/条件问题。

**操作**：
1. 选一个 init1 的 5-epoch checkpoint（`timepoint_512_b12_e5_20260706_000411/checkpoint.pth`）
2. 从它 resume，继续跑到 40 epoch
3. 每 10 epoch 算 FID，每 5 epoch 生成 visual report（control 初始化）
4. 对比 epoch 5/10/20/30/40 的 generated：
   - 像素差 control vs generated（期望从 <2 增长到 >10）
   - FID（期望从高降到合理值，如 <50）

**预期结果**：
- 如果 generated 仍 ≈ control → 配对/条件是主因，进入阶段 2
- 如果 generated 开始偏离 control → 欠训练是主因，直接跑满训练即可

**脚本**：
```bash
# Resume from epoch 5, train to 40
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config microalgae_timepoint_512 \
  --batch_size 12 --epochs 40 --use_initial 1 --cfg_scale 0.2 \
  --use_ema --skewed_timesteps --class_drop_prob 0.2 \
  --eval_frequency 5 --fid_samples 512 --compute_fid \
  --resume outputs/runs/microalgae/timepoint_512_b12_e5_20260706_000411/checkpoint.pth \
  --output_dir outputs/runs/microalgae/timepoint_512_resume_e40
```

**判断标准**：
- epoch 40 时，control vs generated 像素差 **> 5** → 有学习
- FID < 80 → 生成质量可接受
- 若仍 <2 → 进入阶段 2

---

### 阶段 2：组学条件扩充（优先级：P1，如果阶段 1 失败）

**目标**：将 4 维条件扩展到包含组学 PCA，增强表型变化信号。

**技术路径**：
1. **线性插值组学到 5 分钟粒度**
   - 现有：9 整数小时点（0, 1, 2, 3, 6, 12, 24, 48, 72h），每点 3 replicate
   - 目标：105 个 5 分钟 bin（1.2-74h）
   - 方法：对每个 PCA 分量独立做分段线性插值（scipy.interpolate.interp1d）
   - 保留 replicate 均值，忽略方差

2. **构建 61 维 embedding**
   - `[cond_light, cond_dark, time_norm, time_bin_h, rna_pca_0~27, prot_pca_0~27]`
   - 更新 `build_timegroup_data.py` 的 `build_timegroup_embeddings()`
   - 更新 config `base_condition_dim: 61`

3. **重新训练（40 epoch）**
   - 保持 `use_initial=1`, `batch_random` 配对不变
   - 对比 4 维 vs 61 维的 FID 和像素差

**预期**：
- 如果 61 维 generated 仍 ≈ control → 随机配对是不可克服的结构性限制
- 如果 61 维有改善 → 条件强度是关键，可继续优化

**实现草案**：
```python
# scripts/build_omics_interpolated_embedding.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

omics = pd.read_csv('/home/shockley/myproject/FusionODE/data/omics_bulk.csv')
# Group by condition, average over replicates
omics_mean = omics.groupby(['condition','time']).mean(numeric_only=True).reset_index()

# For each condition, interpolate all PCA columns to 5-min bins
...
```

---

### 阶段 3：配对策略重审（优先级：P2，仅当阶段 1+2 失败）

**目标**：如果充分训练 + 强条件仍学成恒等，则重审任务定义。

**选项**：
1. **接受群体均值任务**
   - 改任务描述为"预测同条件下 T 小时群体的典型形态"
   - 评估指标从 FID 改为"生成样本的时序趋势是否符合真实群体演化"
   - 可能需要加入 per-time FID 来验证

2. **探索 weak supervision**
   - 用 morphology features（面积、圆度等）作为软约束
   - 鼓励 generated 的形态统计量匹配 target 群体

3. **放弃 control→target 范式**
   - 改为无条件生成（从噪声直接生成 T 小时细胞）
   - 或改为 time-series forecasting（序列预测而非配对）

**暂不展开**：需要阶段 1+2 结果才能判断是否必要。

---

## 清理任务（并行进行）

### 代码/配置清理

**删除的分支**：
- init0（噪声初始化）相关代码和 checkpoint — 已证实产生双细胞/偏心 artifact
- init2（control + noise）— 实验性质，暂无明确优势
- 所有 5-epoch 短跑 checkpoint（20GB+）

**保留唯一路径**：
- `use_initial=1`（control 初始化）
- `pairing_mode=batch_random`（现状）
- `cfg_scale=0.2`

**文档修正**：
- CLAUDE.md 修正 `base_condition_dim: 4`（当前）或 61（阶段 2 后）
- 明确"不是同细胞时序对，是群体平均漂移任务"

### 结果清理

**保留**：
- `outputs/runs/microalgae/timepoint_512_b12_e5_20260706_000411/` — 作为 resume 起点
- 最终的 40-epoch run 和 visual reports

**删除**：
- 所有 `noiseinit` / `init2` 相关 reports（6 个目录，~5MB）
- 所有短跑 runs 的中间 checkpoints（保留 best/last 即可）

### 脚本整合

**新增**：
- `scripts/train_convergence_test.sh` — 阶段 1 的 40-epoch 训练脚本
- `scripts/build_omics_interpolated_embedding.py` — 阶段 2 的组学插值
- `scripts/generate_visual_report.sh` — 标准化的 visual report 生成（control 初始化）

**删除/归档**：
- `scripts/sample_microalgae_checkpoint.py` 中 `--init-mode noise` 分支
- init2 相关的 noise_prob/noise_level 参数（从 train.sh 和 args.py）

---

## 下一步

1. **立即执行**：阶段 1 的 40-epoch 训练（预计 6-12 小时）
2. **等待结果**：根据 epoch 40 的 generated vs control 像素差，决定是否需要阶段 2
3. **并行清理**：删除发散分支的代码和结果

**决策点**：epoch 40 结果出来后，在这个文件里更新"阶段 1 结果"章节，再决定是直接收工还是进入阶段 2。
