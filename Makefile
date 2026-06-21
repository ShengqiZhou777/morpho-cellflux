SHELL := /bin/bash
.SHELLFLAGS := -lc

PYTHON ?= python

.PHONY: data build-perturbmulti build-diet train train-diet interpolate smoke

# ---- data ----
data:
	$(PYTHON) scripts/materialize_data.py --config configs/crispr_hep.yaml

build-perturbmulti:
	$(PYTHON) scripts/build_perturbmulti_data.py

build-diet:
	$(PYTHON) scripts/build_diet_data.py

# ---- training (scripts/train.sh is env-var parameterized) ----
train:
	OUT=outputs/runs/crispr/dev bash scripts/train.sh

train-diet:
	OUT=outputs/runs/diet/dev CONFIG=diet_id DATASET=diet_id bash scripts/train.sh

# ---- figures ----
interpolate:
	CKPT=$(CKPT) OUT=$(OUT) GPU=$(or $(GPU),0) bash scripts/interpolate.sh

# ---- quick 1-GPU sanity run ----
smoke:
	OUT=outputs/runs/smoke NPROC=1 BATCH=4 EPOCHS=1 EVAL_FREQ=1000 FID_SAMPLES=16 \
	  bash scripts/train.sh
