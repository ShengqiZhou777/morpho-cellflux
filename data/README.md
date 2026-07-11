# Data Directory

This project uses a two-level data layout:

```text
raw/        localized source data under raw/microalgae_v1/
processed/  generated artifacts, versioned under processed/microalgae_v1/
```

Do not point configs at external data directories. Copy external source data
under `raw/microalgae_v1/`, then build or reuse files under
`processed/microalgae_v1/`.

See `../docs/DATA.md` for the full data contract.
