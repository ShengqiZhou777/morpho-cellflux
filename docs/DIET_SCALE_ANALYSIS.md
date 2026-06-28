# Diet Dataset Scale Analysis

**Date**: 2026-06-27  
**Question**: Diet has only 2 nutritional treatments (fasted, HFD) + 1 control (adlib). Is the full 335k-cell dataset necessary, or can we train with fewer cells?

## 1. Dataset Overview

- **Total cells**: 335,099
- **Conditions**: 3 (fasted, adlib, hfd)
- **Hepatocyte subpopulations**: 6 (Hep1, Hep2, Hep3, Hep4, Hep5, Hep6)
- **Effective (condition × cluster) groups**: 18

### Per-Condition Cell Counts

| Condition | Cells | % of Total | Role |
|-----------|------:|:----------:|------|
| adlib | 114,494 | 34.2% | negative_control |
| fasted | 114,722 | 34.2% | treated |
| hfd | 105,883 | 31.6% | treated |

### Per-Cluster Cell Counts (Hepatocyte Subpopulations)

| Cluster | Cells | % of Total |
|---------|------:|:----------:|
| Hep1 | 74,522 | 22.2% |
| Hep2 | 67,908 | 20.3% |
| Hep3 | 17,180 | 5.1% |
| Hep4 | 26,667 | 8.0% |
| Hep5 | 39,825 | 11.9% |
| Hep6 | 108,997 | 32.5% |

### Condition × Cluster Distribution

| Condition | Hep1 | Hep2 | Hep3 | Hep4 | Hep5 | Hep6 | Total |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| adlib | 24,707 | 23,996 | 6,593 | 8,309 | 14,691 | 36,198 | 114,494 |
| fasted | 26,943 | 23,290 | 4,478 | 8,585 | 11,807 | 39,619 | 114,722 |
| hfd | 22,872 | 20,622 | 6,109 | 9,773 | 13,327 | 33,180 | 105,883 |

**Rarest (condition, cluster) group**: 4,478 cells

## 2. Hepatocyte Diversity Analysis

The 6 hepatocyte subpopulations (Hep1-Hep6) represent distinct metabolic zones and functional states within the liver. Each may respond differently to dietary intervention. When downsampling, we must ensure every (condition, cluster) group retains sufficient representation.

### Cluster Entropy per Condition

Higher entropy = more even distribution across Hep clusters:

| Condition | Cluster Entropy | % of Max (ln 6) |
|-----------|:---------------:|:---------------:|
| adlib | 1.641 | 92% |
| fasted | 1.586 | 89% |
| hfd | 1.659 | 93% |

With 3 nutritional conditions × 6 clusters, the model sees 18 distinct (condition, cluster) contexts. The effective diversity is 91% of maximum (perfectly uniform cluster distribution within each condition).

## 3. Scaling Analysis

For each subset size, we project the minimum cells per (condition, cluster) group assuming proportional stratified sampling. Groups falling below 10 cells risk being inadequately represented.

| Subset Size | Min Cells/Group | Groups <10 | Groups <5 | Verdict |
|------------:|:---------------:|:----------:|:---------:|---------|
| 2,000 | 27 | 0 | 0 | ✅ All groups well-represented |
| 5,000 | 67 | 0 | 0 | ✅ All groups well-represented |
| 10,000 | 134 | 0 | 0 | ✅ All groups well-represented |
| 25,000 | 334 | 0 | 0 | ✅ All groups well-represented |
| 50,000 | 668 | 0 | 0 | ✅ All groups well-represented |
| 100,000 | 1336 | 0 | 0 | ✅ All groups well-represented |
| 150,000 | 2004 | 0 | 0 | ✅ All groups well-represented |
| 200,000 | 2673 | 0 | 0 | ✅ All groups well-represented |
| 335,099 | 4478 | 0 | 0 | ✅ All groups well-represented |

## 4. Cluster Representation Stability

Measuring how much the cluster distribution deviates from the full population at different subsample sizes. Lower CV = more stable representation.

### adlib

| Subset Size | Cluster Dist CV | Available |
|------------:|:---------------:|:---------:|
| 100 | 0.019338 | 114,494 |
| 500 | 0.010086 | 114,494 |
| 1,000 | 0.005935 | 114,494 |
| 5,000 | 0.002489 | 114,494 |
| 10,000 | 0.001911 | 114,494 |
| 50,000 | 0.000599 | 114,494 |

### fasted

| Subset Size | Cluster Dist CV | Available |
|------------:|:---------------:|:---------:|
| 100 | 0.016782 | 114,722 |
| 500 | 0.008098 | 114,722 |
| 1,000 | 0.005860 | 114,722 |
| 5,000 | 0.002546 | 114,722 |
| 10,000 | 0.001878 | 114,722 |
| 50,000 | 0.000680 | 114,722 |

### hfd

| Subset Size | Cluster Dist CV | Available |
|------------:|:---------------:|:---------:|
| 100 | 0.018835 | 105,883 |
| 500 | 0.008353 | 105,883 |
| 1,000 | 0.005468 | 105,883 |
| 5,000 | 0.002676 | 105,883 |
| 10,000 | 0.001753 | 105,883 |
| 50,000 | 0.000639 | 105,883 |

## 5. Recommendation

### Key Findings

1. **Effective diversity**: Despite only 3 conditions, the model must learn 18 distinct (condition × cluster) contexts, each with potentially different morphological responses.
2. **Rarest group**: The smallest (condition, cluster) combination has only 4,478 cells in the full dataset. Any downsampling amplifies this sparsity.
3. **Minimum viable size**: 2,000 cells is the **absolute minimum** (all groups ≥ 5 cells). At this scale, rare subpopulations are barely represented.
4. **Recommended minimum**: 2,000 cells is the **safe minimum** (all 18 groups ≥ 10 cells). This preserves cluster diversity while reducing data volume by 99%.
5. **Full dataset value**: The 335k dataset provides 10-50× the cells needed for basic representation. The extra data primarily reduces sampling noise and better captures natural biological variability within each (condition, cluster) group. Whether this matters depends on the model's sensitivity to within-group variance.

### Practical Suggestion

For **development and hyperparameter tuning**, use the **50k** subset (~15% of full, all groups adequately represented, faster iteration). For **final paper results**, use the **full 335k** dataset to capture the full biological variability and maximize statistical power.

To validate this recommendation empirically, run a scaling experiment: train identical models on 50k, 100k, 200k, and full subsets, then compare PGC and FID. If metrics plateau at 100k, the full dataset's marginal benefit is limited.

### Pre-built Subset Indices

| Subset | Path | Use Case |
|--------|------|----------|
| 2k | `data/processed/diet/index_diet_2k.csv` | Quick debug / CI smoke test |
| 5k | `data/processed/diet/index_diet_5k.csv` | Fast validation (paper quick_validate) |
| 50k | `data/processed/diet/index_diet_50k.csv` | Development & hyperparameter tuning |
| 100k | `data/processed/diet/index_diet_100k.csv` | Scaling experiment mid-point |
| 200k | `data/processed/diet/index_diet_200k.csv` | Scaling experiment near-full |
| 335k | `data/processed/diet/index_diet.csv` | Full dataset (paper results) |
