.SHELLFLAGS := -lc
SHELL := /bin/bash

CONDA := source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate pmf

.PHONY: data build-perturbmulti build-diet train train-diet interpolate smoke

# ---- data ----
data:
	$(CONDA) && python scripts/materialize_data.py --config configs/crispr_hep.yaml

build-perturbmulti:
	$(CONDA) && python scripts/build_perturbmulti_data.py

build-diet:
	$(CONDA) && python scripts/build_diet_data.py

# ---- training (scripts/train.sh is env-var parameterized) ----
train:
	$(CONDA) && OUT=outputs/perturbmulti_run bash scripts/train.sh

train-diet:
	$(CONDA) && OUT=outputs/diet_run CONFIG=diet_id DATASET=diet_id bash scripts/train.sh

# ---- figures ----
interpolate:
	$(CONDA) && CKPT=$(CKPT) OUT=$(OUT) GPU=$(or $(GPU),0) bash scripts/interpolate.sh

# ---- quick 1-GPU sanity run ----
smoke:
	$(CONDA) && OUT=outputs/smoke NPROC=1 BATCH=4 EPOCHS=1 EVAL_FREQ=1000 FID_SAMPLES=16 \
	  bash scripts/train.sh
