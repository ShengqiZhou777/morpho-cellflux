# repo-structure Specification

## Purpose

Define the converged contract for the PhenoFlux repository after migration to the microalgae era: a committed microalgae baseline, generated data excluded from version control, a single documented active training path, a single source of truth for the conditioning dimension, and isolation of historical and scratch material. This capability governs repository structure and hygiene only; it does not alter training, inference, or evaluation runtime behavior.

## Requirements

### Requirement: Committed microalgae baseline
The repository HEAD SHALL reflect the microalgae era. The full migration from the CRISPR/Diet era SHALL be committed as a set of reviewable logical commits, so that `git diff` against HEAD reflects microalgae-internal changes rather than the CRISPR-era baseline.

#### Scenario: HEAD reflects microalgae era
- **WHEN** a developer runs `git log -1` after convergence
- **THEN** the latest commit describes microalgae migration work, not CRISPR/Diet
- **AND** `git status --short` shows no pending deletion of the migration files

#### Scenario: Migration is split into reviewable commits
- **WHEN** the migration is committed
- **THEN** it is expressed as multiple logical commits (removals, core, eval+configs, scripts+docs)
- **AND** no single commit mixes the -8757-line removal wall with thousands of new files

### Requirement: Generated data excluded from version control
Generated artifacts SHALL be excluded from version control. This includes synthetic validation datasets, root-level generated images, and scratch JSON.

#### Scenario: Synthetic validation data is ignored
- **WHEN** `data/synthetic_validation/` contains generated images
- **THEN** `git status --porcelain` does NOT list those images as untracked or staged

#### Scenario: No generated artifact is committed
- **WHEN** any convergence commit is created
- **THEN** the committed file set contains no generated image, checkpoint, or scratch JSON

### Requirement: Single active training path
The repository SHALL expose exactly one documented current training path, discoverable via the chain `README.md → docs/DATA.md → configs/README.md → scripts/train.sh`. Additional lanes (e.g. field) SHALL be explicitly labeled active or archived.

#### Scenario: One obvious path for new work
- **WHEN** a new contributor reads `README.md`
- **THEN** it points to a single current config and entry script
- **AND** every config under `configs/` is labeled active or archived in `configs/README.md`

### Requirement: Condition dimension truth is consistent
The conditioning dimension SHALL have a single source of truth. The value declared in the active config, referenced in project documentation, and encoded in the embedding data filename SHALL all agree.

#### Scenario: Dimension agrees across config, docs, and data
- **WHEN** the active config declares `base_condition_dim: 62`
- **THEN** the embedding file is named `embedding_62d.csv`
- **AND** `CLAUDE.md` and any active docs state 62, not 4 / 61 / 92

### Requirement: Historical and scratch material is isolated
Historical campaign material and scratch artifacts SHALL live under `archive/` or `docs/`, not at the repository root. The repository root SHALL contain only standard project files.

#### Scenario: Root contains only standard files
- **WHEN** a developer lists the repository root after convergence
- **THEN** narrative campaign logs live under `docs/experiments/`
- **AND** scratch scripts, images, and temporary JSON are removed or relocated under `data/reports/`

#### Scenario: Retired plans are marked
- **WHEN** a superseded planning doc (e.g. `CLEANUP_PLAN.md`, `ACTION_PLAN.md`) remains for provenance
- **THEN** it is relocated or explicitly marked as retired, so it no longer impersonates current state
