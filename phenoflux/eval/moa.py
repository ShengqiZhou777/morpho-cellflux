import argparse
from collections import defaultdict
import os
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from phenoflux.training.dataloader import CellDataLoader_Eval
import torchvision.transforms as T
import numpy as np
import pandas as pd
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from sklearn.metrics import f1_score

class CustomTransform:
    """Class for scaling and resizing an input image, with optional augmentation and normalization."""

    def __init__(self, augment=False, normalize=False, dim=0):
        self.augment = augment
        self.normalize = normalize
        self.dim = dim

    def __call__(self, X):
        random_noise = torch.rand_like(X)  # Generate random noise
        X = (X + random_noise) / 255.0  # Scale to 0-1 range

        t = []
        if self.normalize:
            num_channels = X.shape[self.dim]
            mean = [0.5] * num_channels
            std = [0.5] * num_channels
            t.append(T.Normalize(mean=mean, std=std))

        if self.augment:
            t.append(T.RandomHorizontalFlip(p=0.3))
            t.append(T.RandomVerticalFlip(p=0.3))

        trans = T.Compose(t)
        return trans(X)

class MOAClassifier(nn.Module):
    def __init__(self, num_classes, device):
        super(MOAClassifier, self).__init__()

        self.feature_extractor = FrechetInceptionDistance(normalize=True).to(device=device, non_blocking=True)
        for param in self.feature_extractor.inception.parameters():
            param.requires_grad = False  # Freeze FID Inception parameters


        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        ).to(device)

    def forward(self, x):
        features = (x * 255).byte()
        features = self.feature_extractor.inception(features)
        outputs = self.classifier(features)
        return outputs

def save_checkpoint(model, optimizer, epoch, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(state, save_path)
    print(f"Checkpoint saved at {save_path}")

def load_checkpoint(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {load_path}, starting from epoch {start_epoch}")
    return start_epoch

def read_img_from_path(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
    return img

def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    save_path="checkpoint_ood.pth",
    label_map=None,
):
    model.to(device)
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_real_ctrl, x_real_trt = batch['X']
            images = torch.clamp(x_real_trt * 0.5 + 0.5, min=0.0, max=1.0).to(device)
            labels = batch['mols'].long().to(device)
            if label_map is not None:
                labels = label_map[labels]
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100.*correct/total:.2f}%")

        save_checkpoint(model, optimizer, epoch, save_path)

def evaluate_model(model, dataloader, device, id2y, label_map=None):
    model.eval()
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_real_ctrl, x_real_trt = batch['X']
            images = torch.clamp(x_real_trt * 0.5 + 0.5, min=0.0, max=1.0).to(device)
            labels = batch['mols'].long().to(device)
            if label_map is not None:
                labels = label_map[labels]
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    print(f"Test Accuracy: {100. * correct / total:.2f}%")

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    print(f"Weighted-F1 Score: {weighted_f1:.4f}")

    print("\nPer-Class Accuracy:")
    for class_id in class_total:
        acc = 100. * class_correct[class_id] / class_total[class_id]
        print(f"Class {id2y[class_id]}: {acc:.2f}%, Total: {class_total[class_id]}")

def evaluate_generated_image(
    model,
    img_root_path,
    device,
    mol2id,
    per_class_cap=None,
    seed=0,
    out_json=None,
    label_map=None,
    id2label=None,
    label_mode="molecule",
):
    """Classify GENERATED images laid out as <img_root_path>/<class>/*.png.

    Imagefolder-driven (NOT test-loader driven): the true label is the folder/class, so it
    works with any matched-N generated subset and never assumes one generation per test cell.
    Mirrors baselines/compute_image_metrics.py's per-condition design so every method is scored
    identically (same InceptionV3 classifier, same per-class cap).
    """
    model.eval()
    id2mol = {v: k for k, v in mol2id.items()}
    if id2label is None:
        id2label = id2mol
    label_map_np = None
    if label_map is not None:
        label_map_np = label_map.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    root = Path(img_root_path)
    classes = [c for c in mol2id if (root / c).is_dir()]
    if not classes:
        raise SystemExit(f"no class subdirs of {root} match mol2id {list(mol2id)}")

    class_correct, class_total = defaultdict(int), defaultdict(int)
    all_labels, all_preds = [], []
    bs = 64
    for cls in classes:
        mol_label = mol2id[cls]
        label = int(label_map_np[mol_label]) if label_map_np is not None else mol_label
        files = sorted((root / cls).glob("*.png"))
        if per_class_cap and len(files) > per_class_cap:
            files = [files[i] for i in sorted(rng.permutation(len(files))[:per_class_cap])]
        with torch.no_grad():
            for i in range(0, len(files), bs):
                imgs = torch.stack(
                    [read_img_from_path(str(f)) for f in files[i : i + bs]]
                ).to(device) / 255.0
                preds = model(imgs).max(1)[1].cpu().numpy()
                for p in preds:
                    all_preds.append(int(p))
                    all_labels.append(label)
                    class_total[label] += 1
                    class_correct[label] += int(int(p) == label)

    acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted"))
    per_class = {
        id2label[c]: {"acc": 100.0 * class_correct[c] / class_total[c], "n": class_total[c]}
        for c in class_total
    }
    print(f"Test Generated Image from: {img_root_path}")
    print(f"Overall Accuracy: {acc:.2f}%  Macro-F1: {macro_f1:.4f}  Weighted-F1: {weighted_f1:.4f}")
    print("Per-Class Accuracy:")
    for name, d in per_class.items():
        print(f"  Class {name}: {d['acc']:.2f}%, Total: {d['n']}")
    result = {
        "img_root_path": str(img_root_path),
        "label_mode": label_mode,
        "moa_acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "n": len(all_labels),
        "per_class_cap": per_class_cap,
    }
    if out_json:
        Path(out_json).write_text(json.dumps(result, indent=2))
        print(f"-> wrote {out_json}")
    return result


def compute_class_weights(dataset, mol2id, num_classes, device, label_map=None):
    """Inverse-frequency weights for molecule or mapped program labels."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    label_map_cpu = label_map.detach().cpu() if label_map is not None else None
    for mol in dataset.mols["trt"]:
        mol_id = int(mol2id[mol])
        label = int(label_map_cpu[mol_id]) if label_map_cpu is not None else mol_id
        counts[label] += 1
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"cannot compute class weights; empty classes: {missing}")
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def build_label_map(args, mol2id, device):
    if not getattr(args, "label_map_csv", None):
        return None, {v: k for k, v in mol2id.items()}, "molecule"

    table = pd.read_csv(args.label_map_csv)
    key_col = args.label_map_key
    label_col = args.label_map_label
    if key_col not in table.columns or label_col not in table.columns:
        raise ValueError(
            f"{args.label_map_csv} must contain columns {key_col!r} and {label_col!r}"
        )

    prog_id_col = next((c for c in table.columns if c.lower() == "program_id"), None)
    if prog_id_col is not None:
        table = table[table[prog_id_col].astype(int) >= 0].copy()
        table["_label_id"] = table[prog_id_col].astype(int)
    else:
        names = sorted(table[label_col].dropna().astype(str).unique())
        label2id = {name: i for i, name in enumerate(names)}
        table["_label_id"] = table[label_col].astype(str).map(label2id).astype(int)

    by_key = table.drop_duplicates(key_col).set_index(key_col)
    missing = [mol for mol in mol2id if mol not in by_key.index]
    if missing:
        raise ValueError(
            f"label map {args.label_map_csv} is missing {len(missing)} classes from the dataloader: "
            f"{missing[:20]}"
        )

    label_map = torch.full((len(mol2id),), -1, dtype=torch.long, device=device)
    for mol, mol_id in mol2id.items():
        label_map[mol_id] = int(by_key.loc[mol, "_label_id"])
    if torch.any(label_map < 0):
        raise ValueError("label map contains unmapped classes")

    id2label = (
        table[["_label_id", label_col]]
        .drop_duplicates("_label_id")
        .sort_values("_label_id")
        .set_index("_label_id")[label_col]
        .astype(str)
        .to_dict()
    )
    expected = set(range(len(id2label)))
    observed = set(id2label)
    if observed != expected:
        raise ValueError(f"label ids must be contiguous from 0; observed {sorted(observed)}")
    return label_map, id2label, label_col


# Main function
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datamodule = CellDataLoader_Eval(args)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    label_map, id2label, label_mode = build_label_map(args, datamodule.mol2id, device)
    num_classes = len(id2label)  # Default = CPD_NAME classes; with label map = paper program classes.

    model = MOAClassifier(num_classes=num_classes, device=device)
    class_weights = None
    if getattr(args, "class_balanced_loss", False):
        class_weights = compute_class_weights(
            datamodule.training_set,
            datamodule.mol2id,
            num_classes,
            device,
            label_map=label_map,
        )
        print(f"Using class-balanced CE weights: {class_weights.detach().cpu().numpy().round(4).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    if Path(args.ckpt_path).exists():
        start_epoch = load_checkpoint(model, optimizer, args.ckpt_path, device)
    if args.mode == 'train':
        train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_epochs=args.epochs,
            save_path=args.ckpt_path,
            label_map=label_map,
        )

        evaluate_model(model, test_loader, device, id2label, label_map=label_map)
    elif args.mode == 'eval':
        assert args.img_root_path is not None, "Image root path is required for evaluation"
        evaluate_generated_image(model, args.img_root_path, device, datamodule.mol2id,
                                 per_class_cap=args.gen_cap, seed=args.seed, out_json=args.out_json,
                                 label_map=label_map, id2label=id2label, label_mode=label_mode)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root_path', type=str, default=None, help='Image root for results')
    parser.add_argument('--ckpt_path', type=str, default='checkpoint.pth', help='Model path')
    parser.add_argument('--mode', type=str, default='eval', help='Mode: eval or train')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs for the MoA classifier')
    parser.add_argument('--config_path', type=str, default='../configs/bbbc021_all.yaml', help='Config path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--iter_ctrl', type=bool, default=False, help='Iter ctrl')
    parser.add_argument('--pin_mem', type=bool, default=True, help='Pin mem')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--gen-cap', dest='gen_cap', type=int, default=None, help='Per-class cap on generated images (matched-N)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for subsampling generated images')
    parser.add_argument('--out_json', type=str, default=None, help='Where to write the MoA result json')
    parser.add_argument('--label-map-csv', dest='label_map_csv', type=str, default=None, help='Optional CSV mapping CPD_NAME to evaluation labels')
    parser.add_argument('--label-map-key', dest='label_map_key', type=str, default='target_gene', help='Key column in --label-map-csv')
    parser.add_argument('--label-map-label', dest='label_map_label', type=str, default='program', help='Label column in --label-map-csv')
    parser.add_argument('--class-balanced-loss', dest='class_balanced_loss', action='store_true', help='Use inverse-frequency class weights for classifier training')
    cli_args = parser.parse_args()
    yaml_config = load_yaml_config(cli_args.config_path)
    yaml_config.update(vars(cli_args))
    args = SimpleNamespace(**yaml_config)
    main(args)
