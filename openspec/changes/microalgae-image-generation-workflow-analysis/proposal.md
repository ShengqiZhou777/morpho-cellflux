# 微藻图像生成工作流程分析

## 概述

PhenoFlux 使用 **Flow Matching + Conditional UNet** 实现微藻表型图像的条件生成，从 control 状态预测 treated 状态的细胞形态变化。

## 核心技术栈

- **模型**: 标准 UNet (无分子先验模块)
- **训练方法**: Conditional Flow Matching
- **数据类型**: RGB 显微镜图像 (3通道)
- **条件信息**: 时间点 + 组学特征嵌入 (61维/92维)
- **框架**: PyTorch + DDP + Flow Matching库

---

## 完整工作流程

### 1. 数据准备阶段

#### 数据类型
**单细胞级** (`microalgae_base.yaml`):
- 图像: 128×128 RGB 裁剪
- 来源: `../../FusionODE/data/CROPS_RAW_SCALE`
- 条件: 61维嵌入 (时间 + RNA PCA + 蛋白质 PCA)
- 数据量: ~1.4M 细胞对

**视野级** (`microalgae_field_base.yaml`):
- 图像: 原始显微镜视野 (可变尺寸)
- 来源: `../../FusionODE/data/TIMECOURSE`
- 条件: 92维嵌入 (形态统计 + 组学 PCs)
- 数据量: ~46K 视野对

#### 数据加载流程
```
CSV索引 (data/processed/index.csv)
    ↓
read_files_pert() → 读取图像对 (ctrl, trt)
    ↓
batch_random 配对 → 同批次内随机配对
    ↓
数据增强 (augment_train=True)
    ↓
返回 batch: (x_ctrl, x_trt, condition_emb)
```

**关键文件**:
- `phenoflux/training/data_utils.py`: `read_files_pert()`
- `phenoflux/training/dataloader.py`: `CellDataset`, `CellDatasetFold`

---

### 2. 模型架构

#### 标准 UNet (简化版)
```python
# phenoflux/models/configs.py
MODEL_CONFIGS = {
    "phenoflux": {
        "in_channels": 3,          # RGB
        "out_channels": 3,         # RGB
        "model_channels": 128,     # 基础通道
        "num_res_blocks": 4,
        "channel_mult": [2, 2, 2],
        "condition_dim": 61/92,    # 从 YAML 设置
    }
}
```

#### Forward 流程
```python
# phenoflux/models/unet.py: UNetModel.forward()

def forward(x, timesteps, extra):
    # 1. 时间嵌入
    emb = time_embed(timestep_embedding(timesteps))  # t → 512维
    
    # 2. 条件嵌入融合
    cond = extra.get("concat_conditioning")  # 61/92维
    if cond is not None:
        mol_embedding = mol_embed_transform(cond)  # 61/92 → 512
        emb = emb + mol_embedding  # 直接相加
    
    # 3. UNet 编码器
    for module in input_blocks:
        h = module(h, emb)
        hs.append(h)  # 保存 skip connections
    
    # 4. 中间层
    h = middle_block(h, emb)
    
    # 5. UNet 解码器
    for module in output_blocks:
        h = cat([h, hs.pop()], dim=1)  # 跳跃连接
        h = module(h, emb)
    
    # 6. 输出投影
    return out(h)  # → 3ch RGB
```

**关键特点**:
- ✅ 条件通过 `mol_embed_transform` 投影后加到时间嵌入
- ✅ **无** MSA/PCD/MGFM 等分子先验模块
- ✅ 标准 skip connections

---

### 3. 训练流程 (Flow Matching)

#### 核心循环
```python
# phenoflux/training/train_loop.py: my_train_one_epoch()

for batch in dataloader:
    x_ctrl, x_trt, condition_emb = batch  # control, treated, 条件
    
    # 1. 采样时间步 t
    t = skewed_timestep_sample(batch_size, device)
    # 使用偏斜分布: P_mean=-1.2, P_std=1.2
    
    # 2. 构造 flow path
    if use_initial == 1:
        x_0 = x_ctrl  # 从 control 开始（关键！）
    else:
        x_0 = randn(...)  # 从噪声开始
    
    # 线性插值: x_t = (1-t)*x_0 + t*x_1
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_trt)
    x_t = path_sample.x_t      # 插值图像
    u_t = path_sample.dx_t     # 速度场目标 (dx/dt)
    
    # 3. Classifier-Free Guidance: 随机丢弃条件
    if random() < class_drop_prob:  # 默认 0.2
        conditioning = {}  # 无条件路径
    else:
        conditioning = {"concat_conditioning": condition_emb}
    
    # 4. 模型预测速度场
    pred = model(x_t, t, extra=conditioning)
    
    # 5. 损失计算
    loss = (pred - u_t).pow(2).mean()  # MSE on velocity
    
    # 6. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Flow Matching 原理
```
时间 t=0          t=中间           t=1
  ↓               ↓                ↓
x_ctrl -------> x_t ---------> x_trt
(control)     (插值)        (treated)
  
目标: 学习速度场 u_t = dx/dt
使用 ODE: dx/dt = model(x_t, t, condition)
```

**关键设计**:
1. **use_initial=1**: 从 control 图像开始（非噪声）
   - 保留细胞结构
   - 只学习 perturbation 变化

2. **Skewed timestep**: 更多关注中间阶段
   - t ~ LogNormal(P_mean=-1.2, P_std=1.2)

3. **CFG**: 训练时 20% 概率丢弃条件
   - 学习条件 vs 无条件预测
   - 推理时混合增强生成质量

---

### 4. 推理/生成流程

#### 评估循环
```python
# phenoflux/training/eval_loop.py: eval_model()

# 1. 准备数据
x_ctrl = batch["ctrl_image"]
condition_emb = datamodule.z_emb[target_condition]

# 2. 构造条件
eval_extra = {"concat_conditioning": condition_emb}

# 3. 初始状态
if use_initial == 1:
    x_0 = x_ctrl  # 从 control 开始
else:
    x_0 = randn(...)

# 4. CFG 包装模型
cfg_scaled_model = CFGScaledModel(
    model,
    cfg_scale=0.2  # 混合比例
)

# 5. ODE 求解
samples = ode_solve(
    cfg_scaled_model,
    x_0,
    t_span=[0, 1],
    extra=eval_extra,
    method='dopri5'  # Runge-Kutta
)

# 6. 保存生成图像
save_fid_samples(samples, epoch, condition)
```

#### CFG 缩放机制
```python
class CFGScaledModel:
    def forward(self, x, t, extra):
        # 条件预测
        pred_cond = model(x, t, extra=extra)
        
        # 无条件预测
        pred_uncond = model(x, t, extra={})
        
        # 线性混合
        # pred = uncond + scale * (cond - uncond)
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)
```

**效果**: `cfg_scale > 0` 增强条件引导，生成更贴合目标条件的图像

---

### 5. 评估指标

#### 图像质量 (phenoflux/eval/fid.py)
- **FIDo** (Open-set FID): 全局特征距离
- **FIDc** (Closed-set FID): 每个条件内的距离
- **KIDo/c**: Kernel Inception Distance

#### 生物学指标 (phenoflux/eval/aggregate.py)
- **PGC** (Phenotypic Gap Closure): 表型缺口闭合率
  - 衡量生成图像与真实 treated 的接近程度
- **方向相关性**: 预测变化方向 vs 真实方向
- **符号一致性**: 变化方向的符号匹配
- **Pearson 相关**: 通道级强度相关

---

## 端到端数据流图

```
┌─────────────────────────────────────────────────────────┐
│ 训练阶段                                                 │
└─────────────────────────────────────────────────────────┘

CSV索引 → read_files_pert()
  ↓
(ctrl_img, trt_img, condition_emb)
  ↓
batch_random 配对 (同批次内随机)
  ↓
采样 t ~ skewed_timestep()
  ↓
构造 flow: x_0=ctrl → x_1=trt
插值: x_t = (1-t)*x_0 + t*x_1
  ↓
UNet(x_t, t, condition_emb) → 预测速度 pred
  ↓
loss = MSE(pred, dx/dt)
  ↓
反向传播 + 优化


┌─────────────────────────────────────────────────────────┐
│ 推理阶段                                                 │
└─────────────────────────────────────────────────────────┘

加载 ctrl_img + target_condition
  ↓
x_0 = ctrl_img (use_initial=1)
  ↓
ODE 求解: t=0→1
使用 CFGScaledModel(cfg_scale=0.2)
  ↓
生成 treated 图像
  ↓
评估: FID + 生物学指标
  ↓
保存到 fid_samples/
```

---

## 关键约束集合

### 硬约束 (技术限制)

1. **数据格式约束**
   - 输入: RGB 图像 (3通道)
   - 条件: 固定维度嵌入 (61维/92维)
   - 配对: control-treated 成对数据

2. **模型架构约束**
   - UNet 输入/输出: 3 通道
   - 条件嵌入维度: 必须与 `base_condition_dim` 匹配
   - 无分子先验模块 (MSA/PCD/MGFM 已移除)

3. **训练约束**
   - Flow matching: 需要成对数据 (x_0, x_1)
   - CFG: 需要训练时随机丢弃条件
   - DDP: 多卡训练时需要 `find_unused_parameters=False`

4. **内存约束**
   - 单细胞级: batch_size=32 (2×32GB GPU)
   - 视野级: batch_size=16 (图像更大)
   - 需要 AMP (自动混合精度) 节省内存

### 软约束 (设计选择)

1. **配对策略**: `batch_random`
   - 适合连续时间序列
   - 保留批次协变量结构
   - 其他选项: `merfish_nn`, `cluster_match`

2. **初始状态**: `use_initial=1`
   - 从 control 开始（推荐）
   - 保留细胞形态
   - 替代: 从噪声开始 (`use_initial=0`)

3. **时间采样**: `skewed_timesteps=True`
   - 偏向中间阶段
   - 更稳定的训练
   - 替代: 均匀采样

4. **数据增强**: `augment_train=True`
   - 随机翻转、旋转
   - 增强泛化能力
   - 由 `data_transform.py` 实现

### 依赖关系

1. **模块依赖**
   ```
   train.py
     ├─ models/configs.py (instantiate_model)
     ├─ training/dataloader.py (CellDataLoader)
     ├─ training/train_loop.py (my_train_one_epoch)
     └─ training/eval_loop.py (eval_model)
   
   dataloader.py
     └─ training/data_utils.py (read_files_pert)
   
   train_loop.py
     └─ flow_matching.path (CondOTProbPath)
   ```

2. **配置依赖**
   - YAML 配置 → `args.py` → 模型/数据加载
   - `base_condition_dim` 必须与嵌入文件匹配

3. **数据依赖**
   - 图像路径: `image_path` (FusionODE 目录)
   - 索引文件: `data_index_path` (CSV)
   - 嵌入文件: `embedding_path` (CSV, 61/92维)

---

## 潜在风险与缓解

### 风险 1: 内存不足
**表现**: OOM errors, CUDA malloc failures
**缓解**:
- 降低 `batch_size`
- 启用 AMP (`torch.amp.autocast`)
- 设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`

### 风险 2: 训练不稳定
**表现**: Loss 震荡, NaN loss
**缓解**:
- 使用 `skewed_timesteps` 偏斜采样
- 降低学习率
- 检查数据归一化 (`normalize=True/False`)

### 风险 3: 生成质量差
**表现**: FID 高, 生物学指标差
**缓解**:
- 增加训练 epochs
- 调整 `cfg_scale` (推荐 0.1-0.3)
- 检查条件嵌入质量

### 风险 4: 数据加载慢
**表现**: GPU 利用率低, 训练慢
**缓解**:
- 增加 `num_workers` (dataloader)
- 预加载图像到内存
- 使用 SSD 存储数据

---

## 成功判据

### 训练成功标准
1. ✅ Loss 稳定下降至 < 0.01
2. ✅ 验证集 FIDc < 50
3. ✅ 无 NaN/Inf loss
4. ✅ GPU 利用率 > 80%

### 生成质量标准
1. ✅ FIDo/FIDc 接近 baseline
2. ✅ PGC > 0.7 (表型缺口闭合 > 70%)
3. ✅ 方向相关性 > 0.8
4. ✅ 视觉检查: 细胞形态合理

### 代码质量标准
1. ✅ 所有测试通过
2. ✅ 模型可保存/加载
3. ✅ 推理速度 < 5s/image (GPU)
4. ✅ 内存占用 < 24GB (单卡)

---

## 快速启动命令

### 单细胞级训练
```bash
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config microalgae_base \
  --batch_size 32 --epochs 20 --use_initial 1 \
  --cfg_scale 0.2 --use_ema --skewed_timesteps \
  --class_drop_prob 0.2 --eval_frequency 5 \
  --output_dir outputs/runs/microalgae/cell_v1
```

### 视野级训练
```bash
torchrun --standalone --nproc_per_node=2 -m phenoflux.train \
  --dataset phenoflux --config microalgae_field_base \
  --batch_size 16 --epochs 20 --use_initial 1 \
  --cfg_scale 0.2 --use_ema --skewed_timesteps \
  --class_drop_prob 0.2 --eval_frequency 5 \
  --output_dir outputs/runs/microalgae/field_v1
```

### 评估
```bash
python phenoflux/eval/fid.py \
  --real-dir outputs/.../fid_samples/epoch-20/real \
  --gen-dir outputs/.../fid_samples/epoch-20/gen \
  --per-condition-cap 500
```

---

## 相关文档

- **项目主文档**: `CLAUDE.md`
- **归档报告**: `TOTAL_CLEANUP_REPORT.md`
- **配置文件**: `configs/microalgae_*.yaml`
- **训练脚本**: `phenoflux/train.py`
- **评估脚本**: `phenoflux/eval/*.py`

---

## 总结

PhenoFlux 微藻图像生成的核心是 **Conditional Flow Matching**:

1. **从 control 状态开始** (use_initial=1)
2. **学习速度场** 预测 treated 状态
3. **条件嵌入** 引导生成过程
4. **CFG** 增强生成质量
5. **ODE 求解** 生成最终图像

**关键优势**:
- ✅ 保留细胞形态结构
- ✅ 条件可控的生成
- ✅ 无需复杂的分子先验模块
- ✅ 端到端可微分

**适用场景**:
- 微藻时间序列表型预测
- 条件图像生成
- 生物学扰动响应建模
