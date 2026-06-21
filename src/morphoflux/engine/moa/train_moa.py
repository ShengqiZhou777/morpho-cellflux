import argparse
from collections import defaultdict
import os
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from morphoflux.engine.training.dataloader import CellDataLoader_Eval
import torchvision.transforms as T
import numpy as np
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

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, save_path="checkpoint_ood.pth"):
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

def evaluate_model(model, dataloader, device, id2y):
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

def evaluate_generated_image(model, img_root_path, device, mol2id, per_class_cap=None, seed=0, out_json=None):
    """Classify GENERATED images laid out as <img_root_path>/<class>/*.png.

    Imagefolder-driven (NOT test-loader driven): the true label is the folder/class, so it
    works with any matched-N generated subset and never assumes one generation per test cell.
    Mirrors baselines/compute_image_metrics.py's per-condition design so every method is scored
    identically (same InceptionV3 classifier, same per-class cap).
    """
    model.eval()
    id2mol = {v: k for k, v in mol2id.items()}
    rng = np.random.default_rng(seed)
    root = Path(img_root_path)
    classes = [c for c in mol2id if (root / c).is_dir()]
    if not classes:
        raise SystemExit(f"no class subdirs of {root} match mol2id {list(mol2id)}")

    class_correct, class_total = defaultdict(int), defaultdict(int)
    all_labels, all_preds = [], []
    bs = 64
    for cls in classes:
        label = mol2id[cls]
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
        id2mol[c]: {"acc": 100.0 * class_correct[c] / class_total[c], "n": class_total[c]}
        for c in class_total
    }
    print(f"Test Generated Image from: {img_root_path}")
    print(f"Overall Accuracy: {acc:.2f}%  Macro-F1: {macro_f1:.4f}  Weighted-F1: {weighted_f1:.4f}")
    print("Per-Class Accuracy:")
    for name, d in per_class.items():
        print(f"  Class {name}: {d['acc']:.2f}%, Total: {d['n']}")
    result = {
        "img_root_path": str(img_root_path),
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


# Main function
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datamodule = CellDataLoader_Eval(args)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    num_classes = len(datamodule.mol2id)  # MoA analog = perturbation class (CPD_NAME), NOT ANNOT (treated/ctrl)

    model = MOAClassifier(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    if Path(args.ckpt_path).exists():
        start_epoch = load_checkpoint(model, optimizer, args.ckpt_path, device)
    id2mol = {v: k for k, v in datamodule.mol2id.items()}
    id2y = datamodule.id2y
    if args.mode == 'train':
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=args.epochs, save_path=args.ckpt_path)

        evaluate_model(model, test_loader, device, id2mol)
    elif args.mode == 'eval':
        assert args.img_root_path is not None, "Image root path is required for evaluation"
        evaluate_generated_image(model, args.img_root_path, device, datamodule.mol2id,
                                 per_class_cap=args.gen_cap, seed=args.seed, out_json=args.out_json)


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
    cli_args = parser.parse_args()
    yaml_config = load_yaml_config(cli_args.config_path)
    yaml_config.update(vars(cli_args))
    args = SimpleNamespace(**yaml_config)
    main(args)
