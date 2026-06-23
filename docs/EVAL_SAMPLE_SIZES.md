# 同类论文的评估样本量 (Eval Sample Sizes)

整理目的：确定 PhenoFlux 评估应采用的每条件生成样本数 N。结论先行——
**当前用 64 张/条件 (总 128) 偏低，应对齐到每条件 ≥500 (MorphGen 下限)，
FID 最好对齐 CellFlux 的 5000 pooled。** 所有方法必须用同一 N，否则不公平
(FID 对 N 敏感，有正偏差)。

所有数字均附原文出处，可直接用于 Setup 节或 rebuttal。

---

## 速查表

| 论文 | 年份 | 指标 | 每条件 / 每扰动评估 N | 数字类型 |
|---|---|---|---|---|
| **MorphGen** | 2025 | FID/KID | **500 生成 vs 500 真实**，<500 增广补齐，3 seed | 评估 N（最可比，per-condition 下限） |
| **CellFlux** | 2025 | FID/KID | **5000 张** (BBBC021, pooled)；RxRx1/JUMP 取 100 扰动类 | 评估 N |
| **IMPA** | 2024 | FID/Coverage/Acc | 整个 held-out test set per drug；跨 drug 平均 + 95% CI | 评估方式（用全 test set，非固定小 N） |
| **PhenDiff** | 2023 | FID/KID/P/R | translation 模式：所有 control→treated 翻译图 vs real treated | 评估方式（无固定小 N） |

可直接引用的"每条件 N"硬数字：**MorphGen 500、CellFlux 5000**。
IMPA/PhenDiff 用的是"全 test set"，不是小批量——同样远大于 64。

---

## 1. MorphGen (arXiv:2510.01298, 2025) — 最可比

> "Scores are computed using **500 generated vs. 500 real images**.
> All experiments are repeated with **three random seeds**."

> "When the real dataset for a given perturbation contained **< 500 examples**,
> we followed [23] and synthetically expanded it using random flips and 90° rotations."

→ per-condition 评估下限 = **500**，不足则增广。最新、最贴合 PhenoFlux 的逐条件评估场景。
链接：https://arxiv.org/abs/2510.01298

## 2. CellFlux (arXiv:2502.09775, ICML 2025) — 主基线

> "FID and KID ... **computed on 5K generated images for BBBC021** and
> **100 randomly selected perturbation classes for RxRx1 and JUMP**;
> we report both overall scores across all samples and conditional scores
> per perturbation class."

数据集规模：
> "98K, 171K, and 424K images ... from 26, 1,042, and 747 perturbation types."

→ FID = **5000 张** (pooled, BBBC021)。我们仓库 `baselines/compute_image_metrics.py`
的 `--n` 默认 5000 即源自此协议（不是新论文）。
链接：https://arxiv.org/abs/2502.09775

## 3. IMPA (Nature Communications 2024, PMC11711326)

> "Batches of true and fake samples are passed through the model and 2048-dimensional
> encodings are extracted from the last pooling layer."（FID 用全 batch，未设小 N）

> "Evaluation metrics comparing generated images with real perturbed images,
> **averaged across different drugs** ... Data are presented as
> **mean values ± 95% confidence intervals**."

> "individual control, transformed control and real perturbation images in the
> test set of a five-drug subset of BBBC021 (**N = 20,313**)."

→ 评估基于整个 held-out test set per drug（万级图），跨 drug 平均 + bootstrap/95% CI。
不是固定小 N，但量级远超 64。
链接：https://www.nature.com/articles/s41467-024-55707-8 ｜ PMC: PMC11711326

## 4. PhenDiff (arXiv:2312.08290, 2023)

训练集规模（§4.1）：
> "BBBC021: ... The training set size of each condition is **1685 images**."
> "Golgi: ... We used **5000 images for each condition**."
> "Organoids: ... **56 images of healthy** organoids and **83 images of diseased**."

评估方式（§4.3 / Table 1）：
> "The metrics are computed using the fake images of treated cells (obtained from the
> translation of images of untreated cells) and the real images of treated cells."

→ translation 模式：评估 N = 参与翻译的 control 图数量（非固定小批）。
论文未给单独的评估 N，引用时以"训练集每条件 1685/5000"作为规模量级参考。
链接：https://arxiv.org/abs/2312.08290

---

## 对 PhenoFlux 的结论

- 当前：fasted 64 + HFD 64 = 128 (`scripts/sweep_diet_moa.sh` 的 `FID_SAMPLES=128` 默认)。
  - FID@64：比同类低 8–80×，几乎不可报（小样本正偏差）。
  - Acc@64：标准误约 ±5%/条件，相邻 epoch/CFG 的几个点差异多为噪声。
- 目标：**每条件 ≥500**（fasted 500 + HFD 500 = 1000）。
  - Acc 标准误 → ±1.8%；FID 达 MorphGen 同档可正经报告。
  - 跑 3 seed 报 均值±std，对齐 MorphGen 协议。
  - diet test set 27K 图，3 条件，采样充足。
- 公平性：baseline (PhenDiff/IMPA/CellFlux) 必须用同一 N 重新评估。
  增大 N 不会"压低 baseline"——FID 全体一起降，Acc 期望不变、只收窄误差棒。

修改入口：`scripts/sweep_diet_moa.sh` 的 `FID_SAMPLES` 128 → 500，对所有方法重生成+重评估。
