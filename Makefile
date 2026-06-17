.SHELLFLAGS := -lc
SHELL := /bin/bash

.PHONY: data

data:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	python scripts/materialize_data.py --config configs/crispr_hep.yaml
