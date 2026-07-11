SHELL := /bin/bash
.SHELLFLAGS := -lc

PYTHON ?= python

.PHONY: data interpolate smoke quick train

# ---- data: build the processed timepoint (128px) and field views from linked raw data ----
data:
	$(PYTHON) scripts/build_microalgae_dataset.py --version microalgae_v1 --views timepoint,field

# ---- omics: interpolate omics PCA -> 62d condition embedding (embedding_62d.csv) ----
interpolate:
	$(PYTHON) scripts/interpolate_omics_to_timepoints.py

# ---- smoke: CPU repo-local validation, no external data required ----
smoke:
	$(PYTHON) scripts/smoke_validate.py

# ---- quick: 1-GPU sanity run of the primary 62d path ----
quick:
	bash scripts/quick_validate.sh microalgae_timepoint_512_62d

# ---- train: full training of the primary 62d config (scripts/train.sh is env-var parameterized) ----
train:
	CONFIG=microalgae_timepoint_512_62d DATASET=phenoflux bash scripts/train.sh
