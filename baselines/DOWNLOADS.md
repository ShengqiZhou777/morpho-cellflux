# External Baseline Downloads

Use these commands from the repository root:

```bash
bash baselines/setup_external_repos.sh
```

This clones code only into:

```text
baselines/external/
  phendiff/
  impa/
  morphodiff/
  stargan/
```

## Required downloads

| Method | Repository | Need original dataset? | Need pretrained weights? | Notes |
|---|---|---:|---:|---|
| PhenDiff | `https://github.com/WarmongeringBeaver/PhenDiff.git` | no | maybe | Train/evaluate on our exported diet/CRISPR panels. If their config requires Stable Diffusion/Diffusers weights, use Hugging Face cache rather than downloading paper datasets. |
| IMPA | `https://github.com/theislab/IMPA.git` | no | no | Use code architecture only; do not download IMPA's BBBC/RxRx processed data for our tables. |
| MorphoDiff | `https://github.com/bowang-lab/MorphoDiff.git` | no | maybe | Use as a named diffusion baseline. Its Stable-Diffusion-style components may expect pretrained VAE/LDM weights depending on config; adapter should document whether training from scratch or fine-tuning. |
| StarGAN | `https://github.com/yunjey/stargan.git` | no | no | Classic multi-domain I2I baseline. Diet is the first target; CRISPR with many classes may be unstable and can be supplement-only. |

## What not to download first

Do not download these unless we explicitly decide to reproduce original-paper
benchmarks in addition to our own:

- IMPA Zenodo preprocessed BBBC/RxRx data.
- CellFlux Hugging Face evaluation index files.
- MorphGen/RxRx1 datasets.
- Original Stable Diffusion checkpoints, unless a selected adapter requires
  them for fine-tuning.

For the BIBM method-comparison table, every method should be trained and
evaluated on the same local data:

```text
configs/diet_id.yaml
configs/perturbmulti_train_id.yaml
data/processed/diet/index_diet.csv
data/processed/perturbmulti/index_train.csv
data/raw/diet_extracted_images/
data/raw/extracted_images/
```

## GPU launch policy

Use one GPU training job at a time across both GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 <method-native-launcher>
```

Do not run PhenDiff and IMPA simultaneously until memory use is measured.

Copy-control is CPU only:

```bash
bash baselines/run_copy_control.sh
```
