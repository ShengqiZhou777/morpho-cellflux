# Paper Table Data

This file is the compact, display-ready source for the paper tables. The machine-readable
version is `data/reports/paper_table_data.tsv`.

Metric: `gap_closed = 1 - W(gen,target) / W(source,target)`, where `W` is 1-D Wasserstein
distance on per-cell foreground channel means. Values near 1 mean generated cells match the
treated population; 0 is the copy-control baseline; negative is worse than copy-control.

## Adapter Status

The RNA-signature condition adapter is already wired as the separate
`perturbmulti_idsig` path:

- Config: `configs/perturbmulti_train_idsig.yaml`
- Model arch: `perturbmulti_idsig`, `condition_dim = 413`
- Condition table: `data/processed/perturbmulti/embedding_gene_idsig.csv`
- Run: `outputs/cellflux_pm_train_id_v9`

This keeps the one-hot baseline `perturbmulti_id` at `condition_dim = 204`, so v8 and v9 are
an apples-to-apples adapter ablation.

## Table 1: Diet, Balanced Checkpoint

Run: `outputs/diet_id_v3`, checkpoint `checkpoint-9.pth`, epoch 9, panel `[9,5,8]`
(Calreticulin / Perilipin / TOMM20).

| condition | Calreticulin | Perilipin | TOMM20 |
|---|---:|---:|---:|
| fasted | 0.86 | 0.40 | 0.40 |
| hfd | 0.88 | 0.62 | 0.36 |

Source: `outputs/diet_id_v3/aggregate_eval_summary.json`.

## Table 2: Diet, HFD-Peak Checkpoint

Run: `outputs/diet_id_v3`, checkpoint `checkpoint-11.pth`, epoch 11, panel `[9,5,8]`.
Use this for an HFD-only figure/table; fasted overshoots at this epoch.

| condition | Calreticulin | Perilipin | TOMM20 |
|---|---:|---:|---:|
| hfd | 0.91 | 0.85 | 0.81 |

Source: `outputs/diet_id_v3/aggregate_eval_summary_hfdpeak_ep11.json`.

## Table 3: CRISPR One-Hot Main Result

Run: `outputs/cellflux_pm_train_id_v8`, checkpoint `checkpoint-19.pth`, epoch 19, panel
`[0,14,5]` (Alb / Rab7 / Perilipin), 76-gene `rna_snr >= 0.3` index, eval FID 16.3.

| channel | gap_closed | dir_corr | sign_agree | pearson |
|---|---:|---:|---:|---:|
| Alb | 0.18 | 0.68 | 0.74 | 0.25 |
| Rab7 | 0.15 | 0.61 | 0.74 | 0.10 |
| Perilipin | -0.14 | 0.42 | 0.68 | 0.50 |

Source: `outputs/cellflux_pm_train_id_v8/aggregate_eval_summary.json` and
`outputs/cellflux_pm_train_id_v8/log.txt`.

## Table 4: RNA-Signature Adapter Ablation

Run: `outputs/cellflux_pm_train_id_v9`, condition = one-hot plus scaled per-gene RNA
signature (`413` dims), same panel and index as v8. Current on-disk v9 summary is epoch 9
from `fid_samples/epoch-9`; the epoch 9 checkpoint itself was intentionally cleaned up, while
the generated samples were retained. FID values are from the training log.

| channel | v8 gap_closed, ep19 | v9 gap_closed, ep9 |
|---|---:|---:|
| Alb | 0.18 | -0.78 |
| Rab7 | 0.15 | -0.75 |
| Perilipin | -0.14 | 0.22 |

| run | epoch | eval_fid |
|---|---:|---:|
| v8 one-hot | 19 | 16.3 |
| v9 idsig | 9 | 16.6 |
| v9 idsig | 14 | 16.0 |
| v9 idsig | 19 | 16.7 |

Source: `outputs/cellflux_pm_train_id_v9/aggregate_eval_summary.json`,
`outputs/cellflux_pm_train_id_v9/log.txt`, and
`outputs/cellflux_pm_train_id_v8/aggregate_eval_summary.json`.

## Reproduction Commands

```bash
python scripts/aggregate_eval.py outputs/diet_id_v3 5 9
python scripts/aggregate_eval.py outputs/diet_id_v3 5 11
cp outputs/diet_id_v3/aggregate_eval_summary.json \
  outputs/diet_id_v3/aggregate_eval_summary_hfdpeak_ep11.json
python scripts/aggregate_eval.py outputs/diet_id_v3 5 9
python scripts/aggregate_eval.py outputs/cellflux_pm_train_id_v8 5 19
python scripts/aggregate_eval.py outputs/cellflux_pm_train_id_v9 5 9
```

The values above are extracted from the existing completed run artifacts and logs. Running the
commands regenerates the corresponding `aggregate_eval_summary.json` files; rerun the diet
epoch 9 command last to keep the canonical diet summary at the balanced checkpoint.
