# Repository Convergence Plan

> **RETIRED (2026-07-11)** — This documents the 2026-07-05 convergence pass and is
> kept for provenance only. It has been superseded by the openspec change
> `converge-microalgae-repo`. Do not treat the paths/configs below as current;
> see `README.md`, `docs/DATA.md`, `configs/README.md`, and `CLAUDE.md` for the
> converged state.

Goal: make the current microalgae project easy to run and reason about while
keeping small historical provenance material archived.

## Scope Lock

Current active scope:

```text
FusionODE microalgae conditional image generation
flow-matching UNet
microalgae_v1/views/timepoint
microalgae_v1/views/field
```

Out of current default scope:

```text
Diet / CRISPR default experiments
Perturb-Multi download flow
MSA / PCD molecular prior claims
external baseline training as the main entry point
```

## Completed In This Pass

- Replaced the top-level README with the current microalgae framing.
- Added `docs/DATA.md` as the raw/processed data-contract source of truth.
- Added `configs/README.md` to explain every active config.
- Added `docs/LEGACY.md` to isolate historical material.
- Updated launcher defaults to favor current microalgae configs.
- Updated active configs so runtime image paths point to `data/raw/...`.
- Added `scripts/prepare_raw_data.sh` to link external FusionODE data into `data/raw/`.
- Moved config expectations from flat `data/processed/*.csv` paths to
  `data/processed/microalgae_v1/...` paths.
- Added `scripts/migrate_processed_layout.sh` for idempotent migration of known
  flat artifacts.
- Added `scripts/build_microalgae_dataset.py` as the publication-style processed
  data rebuild entry point.
- Added candidate public configs, but only `microalgae_timepoint` is active for
  the first runnable target.
- Reduced the default runnable target to `microalgae_timepoint` only.
- Changed the default processed-data builder to build only the `timepoint` view.
- Added `configs/microalgae_smoke.yaml` and `scripts/smoke_validate.py` for
  repo-local CPU validation without external FusionODE data.
- Archived legacy configs, scripts, baseline adapter sources, paper sources, and
  old provenance docs under `archive/legacy_20260705/`.
- Deleted large generated artifacts: CRISPR/DIET raw data, legacy processed
  views, checkpoints, outputs, logs, wandb runs, baseline external repos, paper
  build products, and caches.
- Restored the field lane as a cleaned second active path aligned with
  simulator-like whole-field generation and annotation support.

## Next Verification Pass

Run these before deleting code:

```bash
# Parse/read current docs and configs
python - <<'PY'
from pathlib import Path
import yaml
for path in sorted(Path("configs").glob("microalgae*.yaml")):
    with path.open() as f:
        data = yaml.safe_load(f)
    for key in ["dataset", "dataset_name", "data_index_path", "embedding_path", "base_condition_dim"]:
        assert key in data, f"{path}: missing {key}"
    assert str(data["image_path"]).startswith("data/raw/"), f"{path}: image_path must be under data/raw"
    assert str(data["data_index_path"]).startswith("data/processed/microalgae_v1/"), f"{path}: index path must use microalgae_v1"
    assert str(data["embedding_path"]).startswith("data/processed/microalgae_v1/"), f"{path}: embedding path must use microalgae_v1"
print("configs ok")
PY

python scripts/smoke_validate.py
python scripts/field_smoke_validate.py
```

With external FusionODE data linked and processed:

```bash
bash scripts/quick_validate.sh microalgae_timepoint
```

## Deletion Policy

Only delete after the verification pass:

| Candidate | Action |
| --- | --- |
| stale README snippets | already removed from active README |
| old quick-start defaults | update in scripts rather than delete scripts |
| small legacy source/config/docs | archived under `archive/legacy_20260705/` |
| large data/checkpoints/logs | deleted from working tree |
| smoke fixture | generated under ignored `data/smoke/` |

## Desired End State

The repository should have one obvious path for new work:

```text
README.md -> docs/DATA.md -> configs/README.md -> scripts/train.sh
```

Historical work should be discoverable through:

```text
docs/LEGACY.md
archive_*/
archive/legacy_20260705/
```
