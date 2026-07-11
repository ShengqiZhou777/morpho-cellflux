SHELL := /bin/bash
.SHELLFLAGS := -lc

PYTHON ?= python

.PHONY: data interpolate smoke quick train

# ---- data: build the processed timepoint (128px) and field views from linked raw data ----
data:
	$(PYTHON) scripts/build_microalgae_dataset.py --version microalgae_v1 --views timepoint,field

# ---- omics: interpolate raw gene/protein -> 476-dim condition embedding (embedding_genes.csv) ----
interpolate:
	$(PYTHON) scripts/build_gene_condition.py

# ---- smoke: CPU repo-local validation, no external data required ----
smoke:
	$(PYTHON) scripts/smoke_validate.py

# ---- quick: 1-GPU sanity run of the primary gene path ----
quick:
	bash scripts/quick_validate.sh microalgae_timepoint_512_genes

# ---- train: full training of the primary gene config (scripts/train.sh is env-var parameterized) ----
train:
	CONFIG=microalgae_timepoint_512_genes DATASET=phenoflux bash scripts/train.sh
