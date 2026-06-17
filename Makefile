.SHELLFLAGS := -lc
SHELL := /bin/bash

.PHONY: data train train-long train-scaffold train-puncta-ddp train-lipid-panel-ddp ddp-sanity smoke-train export-preview

data:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	python scripts/materialize_data.py --config configs/crispr_hep.yaml

train:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	torchrun --standalone --nproc_per_node=2 scripts/train_cellflux.py --config configs/train_cellflux.yaml

train-long:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	torchrun --standalone --nproc_per_node=2 scripts/train_cellflux.py --config configs/train_cellflux.yaml --max-steps 10000 --batch-size 8 --output-dir outputs/cellflux_long_10k

train-scaffold:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	torchrun --standalone --nproc_per_node=2 scripts/train_cellflux.py --config configs/train_cellflux_scaffold.yaml --max-steps 5000 --batch-size 8 --output-dir outputs/cellflux_scaffold_5k

train-puncta-ddp:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	torchrun --standalone --nproc_per_node=2 scripts/train_cellflux.py --config configs/train_cellflux_scaffold_mean_puncta.yaml --max-steps 2000 --batch-size 128 --output-dir outputs/cellflux_scaffold_mean_puncta_residual_ddp_2k

train-lipid-panel-ddp:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	torchrun --standalone --nproc_per_node=2 scripts/train_cellflux.py --config configs/train_cellflux_lipid_panel.yaml --max-steps 2000 --batch-size 128 --output-dir outputs/cellflux_lipid_panel_scaffold_ddp_2k

ddp-sanity:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	torchrun --standalone --nproc_per_node=2 scripts/train_cellflux.py --config configs/train_cellflux_lipid_panel.yaml --max-steps 2 --batch-size 128 --limit-train-rows 256 --limit-val-rows 128 --output-dir outputs/ddp_sanity

smoke-train:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	python scripts/train_cellflux.py --device cpu --max-steps 2 --batch-size 1 --limit-train-rows 4 --limit-val-rows 2

export-preview:
	source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
	conda activate pmf && \
	python scripts/export_preview_jpg.py outputs/cellflux_long_10k/previews/step_0010000.npz --sample 0 --out-dir outputs/cellflux_long_10k/jpg_previews
