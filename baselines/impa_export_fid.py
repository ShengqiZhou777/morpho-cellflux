"""Export IMPA generations to the shared CellFlux eval layout."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapter_common import (
    build_pairs,
    channels_from_config,
    load_config,
    panel_array,
    read_index,
    repo_path,
    write_eval_contract,
    write_fid_image,
)


IMPA_ROOT = repo_path("baselines/external/impa")
sys.path.insert(0, str(IMPA_ROOT))

from IMPA.dataset.data_loader import CellDataLoader  # noqa: E402
from IMPA.model import build_model  # noqa: E402


def default_impa_args(config: dict, data_dir: Path, experiment_dir: Path) -> Namespace:
    return Namespace(
        task_name=config.get("task_name", "morphocellflux"),
        img_size=128,
        latent_dim=1024,
        hidden_dim=512,
        style_dim=64,
        stochastic=True,
        z_dimension=8,
        dim_in=64,
        lambda_reg=1,
        lambda_cyc=1,
        lambda_sty=1,
        lambda_ds=1,
        total_epochs=2,
        ds_iter=100000,
        resume_iter=0,
        batch_size=16,
        val_batch_size=8,
        lr=0.0001,
        f_lr=0.0001,
        beta1=0,
        beta2=0.99,
        weight_decay=0.0001,
        num_outs_per_domain=10,
        single_style=True,
        ood_set=None,
        mol_list=None,
        trainable_emb=False,
        dataset_name="bbbc021",
        n_channels=len(channels_from_config(config)),
        num_workers=2,
        seed=42,
        multimodal=False,
        batch_correction=False,
        batch_key=None,
        use_condition_embeddings=False,
        add_controls=False,
        condition_embedding_dimension=None,
        n_mod=1,
        modality_list=["Gene"],
        image_path=str(data_dir / "impa_npy" / "images"),
        data_index_path=str(data_dir / "impa_index.csv"),
        embedding_path=str(data_dir / "impa_embedding.csv"),
        experiment_directory=str(experiment_dir),
        sample_dir="sample",
        checkpoint_dir="checkpoint",
        naming_key="dataset_name",
        resume_dir="",
        augment_train=False,
        normalize=False,
        print_every=10,
        sample_every=1000,
        save_every=500,
        eval_every=500,
        encode_rdkit=True,
        num_layers_mapping_net=1,
        filename="epoch_{epoch:04d}",
        monitor="fid_transformations",
        mode="min",
        save_last=True,
        offline=True,
        project="morphocellflux_impa",
        log_model=False,
        accelerator="gpu",
        log_every_n_steps=10,
    )


def control_tensor(
    image_dir: str,
    sample_key: str,
    channels: list[int],
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    arr = panel_array(image_dir, sample_key, channels).astype("float32")
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x.to(device)


def load_nets_checkpoint(nets, checkpoint: Path, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device)
    for name, module in nets.items():
        if name not in state:
            raise KeyError(f"{checkpoint} missing IMPA module {name!r}; found {sorted(state)}")
        module.module.load_state_dict(state[name])
        module.eval()


def export_impa(
    config_path: Path,
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    benchmark: str,
    split: str,
    seed: int,
    max_samples: int | None,
    device_name: str,
) -> None:
    config = load_config(config_path)
    channels = channels_from_config(config)
    df = read_index(config["data_index_path"])
    targets, trt2ctrl = build_pairs(df, split=split, seed=seed, max_samples=max_samples)

    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    impa_args = default_impa_args(config, data_dir, output_dir / "external_checkpoints")
    datamodule = CellDataLoader(impa_args)
    impa_args.latent_dim = datamodule.latent_dim
    nets = build_model(
        impa_args,
        datamodule.n_mol,
        device,
        multimodal=impa_args.multimodal,
        batch_correction=impa_args.batch_correction,
        modality_list=impa_args.modality_list,
        latent_dim=datamodule.latent_dim,
    )
    load_nets_checkpoint(nets, checkpoint, device)
    embedding_matrix = datamodule.embedding_matrix.to(device)

    torch.manual_seed(seed)
    written = 0
    skipped: list[str] = []
    for _, row in targets.iterrows():
        target_id = str(row["SAMPLE_KEY"])
        condition = str(row["CPD_NAME"])
        if condition not in datamodule.mol2id:
            skipped.append(f"{target_id}: unknown IMPA mol {condition}")
            continue
        control_id = trt2ctrl[target_id]
        try:
            x_ctrl = control_tensor(config["image_path"], control_id, channels, impa_args.img_size, device)
        except FileNotFoundError as exc:
            skipped.append(f"{target_id}: {exc}")
            continue

        y_trg = torch.tensor([datamodule.mol2id[condition]], device=device).long()
        z_emb = embedding_matrix(y_trg)
        if impa_args.stochastic:
            z = torch.randn(x_ctrl.shape[0], impa_args.z_dimension, device=device)
            z_emb = torch.cat([z_emb, z], dim=1)
        with torch.no_grad():
            style = nets.mapping_network(z_emb, y_trg, None)
            _, x_fake = nets.generator(x_ctrl, style)
        image = x_fake.detach().clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
        write_fid_image(output_dir, condition, target_id, image)
        written += 1

    args = {
        "baseline_method": "IMPA",
        "benchmark": benchmark,
        "use_initial": 1,
        "channels": channels,
        "image_path": config["image_path"],
        "data_index_path": config["data_index_path"],
        "embedding_path": config["embedding_path"],
        "dataset_name": config.get("dataset_name", "perturbmulti"),
        "task_name": config.get("task_name"),
        "split": split,
        "seed": seed,
        "max_samples": max_samples,
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "written": written,
        "skipped": skipped,
    }
    write_eval_contract(output_dir, trt2ctrl, args)
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    print(f"IMPA wrote {written} samples to {output_dir / 'fid_samples' / 'epoch-0'}")
    if skipped:
        print(f"IMPA skipped {len(skipped)} samples; see {output_dir / 'skipped.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="IMPA checkpoint/*_nets.ckpt file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_impa(
        config_path=repo_path(args.config),
        data_dir=repo_path(args.data_dir),
        checkpoint=repo_path(args.checkpoint),
        output_dir=repo_path(args.output),
        benchmark=args.benchmark,
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
