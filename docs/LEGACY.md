# Legacy Boundary

This repository contains multiple generations of work. The current default
project is FusionODE microalgae image generation. This file separates active
material from historical material so future cleanup does not accidentally erase
provenance or reproducibility evidence.

## Active By Default

Use these paths for current work:

```text
configs/microalgae_timepoint.yaml
configs/microalgae_field.yaml
configs/microalgae_smoke.yaml
scripts/build_microalgae_dataset.py
scripts/build_field_metadata.py
scripts/build_field_dataset.py
scripts/build_smoke_fixture.py
scripts/smoke_validate.py
scripts/field_smoke_validate.py
scripts/train.sh
scripts/quick_validate.sh
phenoflux/
phenoflux/eval/
data/processed/microalgae_v1/
docs/DATA.md
configs/README.md
```

The active model registry is simplified RGB microalgae UNet configuration in
`phenoflux/models/configs.py`.

## Historical Lineage

The project was adapted from PhenoFlux / CellFlux-style cellular phenotype
transport code. Older documentation and comments may mention:

```text
Diet hepatocyte perturbations
CRISPR gene knockouts
Perturb-Multi
MERFISH marker panels
MSA / PCD molecular priors
```

These concepts explain where the code came from, but they are not current
default entry points unless a new experiment explicitly restores them.

The names `continuous`, `timegroup`, and `leave6` are also treated as legacy
implementation names. Public docs should use `state`, `timepoint`, `field`, and
`leave_time_6h`.

## Archive / Output Material

These paths are not active entry points:

```text
archive/legacy_20260705/
outputs/
data/smoke/
```

Historical source, configs, docs, and provenance material live under
`archive/legacy_20260705/`. Large checkpoints, raw CRISPR/DIET data, generated
outputs, logs, and external baseline checkouts were intentionally deleted in the
storage-pruning pass.

## Cleanup Candidates

Safe cleanup should happen in stages:

1. Update docs and script defaults to current microalgae entry points.
2. Verify current quick validation and one smoke training path.
3. Keep large generated data and checkpoints outside git.
4. Restore archived material only when an explicit experiment needs it.
