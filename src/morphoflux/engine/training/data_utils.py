import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path

# Perturb-Multi hepatocyte multi-pathway panel: channel indices into the
# 18-channel npz (var order: Alb0, polyT1, rRNA2, M6PR3, CathB4, Perilipin5,
# Sqstm1 6, LC3b7, TOMM20 8, Calreticulin9, pS6RP10, ...).
# Legacy fallback panel for configs without an explicit `channels` field.
# Public configs pin their own panels in YAML.
PERTURBMULTI_CHANNELS = [5, 9, 10]

# Precomputed per-condition population-mean 18-channel profiles.
# Used by PhenoFlux MAC/CCM to condition on the TARGET condition's
# canonical molecular state instead of the individual control cell's
# profile (which is confounded with per-cell variation in pseudo-paired data).
_COND_MEAN_PROFILES = None


def _load_perturbmulti(image_path, sample_key, channels=None, return_full_profile=False):
    """Load a Perturb-Multi cell crop: npz['x'] (18,H,W) float[0,1] -> selected
    channels in [-1, 1], already channel-first so no permute/255 transform.
    `channels` (npz indices) is per-config; falls back to PERTURBMULTI_CHANNELS.

    When `return_full_profile` is True, also returns the full (18, H, W) array
    in [0, 1] for use as marker-aware conditioning (Direction A)."""
    channels = channels if channels is not None else PERTURBMULTI_CHANNELS
    full = np.load(Path(image_path) / f"{sample_key}.npz")["x"]
    arr = full[channels]
    img = torch.from_numpy(np.ascontiguousarray(arr)).float()
    result = img * 2.0 - 1.0  # [0,1] -> [-1,1]

    if return_full_profile:
        full_tensor = torch.from_numpy(np.ascontiguousarray(full)).float()
        return result, full_tensor
    return result


def _get_cond_mean_profile(condition_id: int, device=None):
    """Return the population-mean 18-channel profile for a target condition.

    Broadcasts the per-channel scalar means to a (18, 128, 128) spatial tensor
    compatible with the MAC MarkerProfileEncoder Conv2d input.  The constant
    feature maps let the Conv2d extract per-channel features without spatial
    variation — the model learns to condition on the canonical molecular state
    of the TARGET condition rather than the individual (noisy, pseudo-paired)
    control cell.

    Profiles are loaded lazily from ``data/processed/<task>/cond_mean_profiles.npz``.
    """
    global _COND_MEAN_PROFILES
    if _COND_MEAN_PROFILES is None:
        # Try diet first; could be extended with a config-driven path later.
        import os
        candidates = [
            "data/processed/diet/cond_mean_profiles.npz",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "processed", "diet", "cond_mean_profiles.npz"),
        ]
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            raise FileNotFoundError(
                "cond_mean_profiles.npz not found. Run the profile computation step first."
            )
        data = np.load(path)
        _COND_MEAN_PROFILES = {int(k): torch.from_numpy(data[k].astype(np.float32)) for k in data.files}

    mean_18 = _COND_MEAN_PROFILES[int(condition_id)]  # (18,)
    # Broadcast to spatial: (18, 128, 128)
    spatial = mean_18[:, None, None].expand(18, 128, 128).clone()
    if device is not None:
        spatial = spatial.to(device)
    return spatial


class CustomTransform:
    """Class for scaling and resizing an input image, with optional augmentation and normalization."""

    def __init__(self, augment=False, normalize=False, dim=0):
        """
        Initialize the CustomTransform instance.

        Args:
            augment (bool, optional): Whether to apply augmentation (random flips). Defaults to False.
            normalize (bool, optional): Whether to normalize the image. Defaults to False.
            dim (int, optional): Dimension along which the normalization is applied. Defaults to 0.
        """
        self.augment = augment
        self.normalize = normalize
        self.dim = dim

    def __call__(self, X):
        """
        Apply the transformations to the input image.

        Args:
            X (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        # Add random noise and rescale pixels between 0 and 1
        random_noise = torch.rand_like(X)  # Generate random noise
        X = (X + random_noise) / 255.0  # Scale to 0-1 range

        t = []
        # Normalize the input to the range [-1, 1]
        if self.normalize:
            num_channels = X.shape[self.dim]
            mean = [0.5] * num_channels
            std = [0.5] * num_channels
            t.append(T.Normalize(mean=mean, std=std))

        # Perform augmentation steps
        if self.augment:
            t.append(T.RandomHorizontalFlip(p=0.3))
            t.append(T.RandomVerticalFlip(p=0.3))

        trans = T.Compose(t)
        return trans(X)

def read_files_pert(file_names, mols, mol2id, y2id, dose, y, transform, image_path, dataset_name, idx, multimodal, batch, iter_ctrl, channels=None, return_full_profile=False):
    """
    Read and process control and treated batch images.

    Args:
        file_names (dict): Dictionary containing file names for 'ctrl' and 'trt' samples.
        mols (dict): Dictionary containing molecule information for 'ctrl' and 'trt' samples.
        mol2id (dict): Mapping from molecule names to IDs.
        y2id (dict): Mapping from annotation names to IDs.
        dose (dict): Dictionary containing dose information for 'ctrl' and 'trt' samples.
        y (dict): Dictionary containing annotation information for 'ctrl' and 'trt' samples.
        transform (callable): Transformation to apply to the images.
        image_path (str): Path to the image folder.
        dataset_name (str): Name of the dataset.
        idx (int): Index of the sample to retrieve.
        multimodal (bool): Whether the dataset is multimodal.

    Returns:
        dict: Dictionary containing processed images, molecule information, annotation ID, dose, and file names.
    """
    if iter_ctrl:
        # Sample control and treated batches
        img_file_ctrl = file_names["ctrl"][idx]
        idx_trt = np.random.randint(0, len(file_names["trt"]))
        img_file_trt = file_names["trt"][idx_trt]
        idx_ctrl = idx

    else:
        idx_trt = idx
        # Use idx to select trt image and random select a ctrl image from the same batch
        img_file_trt = file_names["trt"][idx_trt]
        batch_trt = batch["trt"][idx_trt]

        ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
        if len(ctrl_indices_same_batch) == 0:
            raise ValueError(f"No control samples found in the same batch as the treated sample (batch: {batch_trt}).")

        idx_ctrl = np.random.choice(ctrl_indices_same_batch)
        img_file_ctrl = file_names["ctrl"][idx_ctrl]

    if dataset_name == "perturbmulti":
        # Direct npz load by cell_id; same-batch ctrl pairing already done above.
        if return_full_profile:
            img_ctrl, full_ctrl = _load_perturbmulti(image_path, img_file_ctrl, channels, return_full_profile=True)
            img_trt, full_trt = _load_perturbmulti(image_path, img_file_trt, channels, return_full_profile=True)
        else:
            img_ctrl = _load_perturbmulti(image_path, img_file_ctrl, channels)
            img_trt = _load_perturbmulti(image_path, img_file_trt, channels)
        # Range-safe flip augmentation. CustomTransform's noise/normalize path assumes
        # [0,255] inputs and is bypassed for perturbmulti, so apply flips directly to the
        # [-1,1] tensors when augmentation is on (train fold + augment_train). Flips do not
        # change pixel values, so the [-1,1] range is preserved; ctrl/trt flip together.
        if getattr(transform, "augment", False):
            if torch.rand(1).item() < 0.3:
                img_ctrl, img_trt = torch.flip(img_ctrl, [-1]), torch.flip(img_trt, [-1])
                if return_full_profile:
                    full_ctrl, full_trt = torch.flip(full_ctrl, [-1]), torch.flip(full_trt, [-1])
            if torch.rand(1).item() < 0.3:
                img_ctrl, img_trt = torch.flip(img_ctrl, [-2]), torch.flip(img_trt, [-2])
                if return_full_profile:
                    full_ctrl, full_trt = torch.flip(full_ctrl, [-2]), torch.flip(full_trt, [-2])
        result = {
            'X': (img_ctrl, img_trt),
            'mols': mol2id[mols["trt"][idx_trt]],
            'y_id': y2id[y["trt"][idx_trt]],
            'file_names': (img_file_ctrl, img_file_trt),
            'idx_trt': idx_trt,
            'idx_ctrl': idx_ctrl,
            'batch': batch_trt,
        }
        if return_full_profile:
            result['marker_profile'] = _get_cond_mean_profile(
                y2id[y["trt"][idx_trt]], device=img_ctrl.device
            )
            # PhenoFlux MAC/CCM: condition on TARGET condition's population-mean
            # 18-channel profile instead of the individual control cell's profile.
            # Pseudo-paired data (same-batch, different cells) means per-cell
            # control profiles are confounded with individual variation; population
            # means capture the canonical molecular state of each condition.
        return result

    # Split files
    file_split_ctrl = img_file_ctrl.split('-')
    file_split_trt = img_file_trt.split('-')

    if len(file_split_ctrl) > 1:
        file_split_ctrl = file_split_ctrl[1].split("_")
        file_split_trt = file_split_trt[1].split("_")
        path_ctrl = Path(image_path) / "_".join(file_split_ctrl[:2]) / file_split_ctrl[2]
        path_trt = Path(image_path) / "_".join(file_split_trt[:2]) / file_split_trt[2]
        file_ctrl = '_'.join(file_split_ctrl[3:]) + ".npy"
        file_trt = '_'.join(file_split_trt[3:]) + ".npy"
    else:
        file_split_ctrl = file_split_ctrl[0].split("_")
        file_split_trt = file_split_trt[0].split("_")
        if dataset_name == "cpg0000":
            path_ctrl = Path(image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}_{file_split_ctrl[2]}"
            path_trt = Path(image_path) / file_split_trt[0] / f"{file_split_trt[1]}_{file_split_trt[2]}"
            file_ctrl = '_'.join(file_split_ctrl[1:]) + ".npy"
            file_trt = '_'.join(file_split_trt[1:]) + ".npy"
        elif dataset_name == "bbbc021":
            path_ctrl = Path(image_path) / file_split_ctrl[0] / f"{file_split_ctrl[1]}"
            path_trt = Path(image_path) / file_split_trt[0] / f"{file_split_trt[1]}"
            file_ctrl = '_'.join(file_split_ctrl[2:]) + ".npy"
            file_trt = '_'.join(file_split_trt[2:]) + ".npy"

    img_ctrl, img_trt = np.load(path_ctrl / file_ctrl), np.load(path_trt / file_trt)
    img_ctrl, img_trt = torch.from_numpy(img_ctrl).float(), torch.from_numpy(img_trt).float()
    img_ctrl, img_trt = img_ctrl.permute(2, 0, 1), img_trt.permute(2, 0, 1)  # Place channel dimension in front of the others
    img_ctrl, img_trt = transform(img_ctrl), transform(img_trt)

    if multimodal:
        y_mod = y["trt"][idx_trt]
        mol = mol2id[y_mod][mols["trt"][idx_trt]]
    else:
        mol = mol2id[mols["trt"][idx_trt]]

    return {
        'X': (img_ctrl, img_trt),
        'mols': mol,
        'y_id': y2id[y["trt"][idx_trt]],
        'dose': dose["trt"][idx_trt],
        'file_names': (img_file_ctrl, img_file_trt),
        'idx_trt': idx_trt,
        'idx_ctrl': idx_ctrl,
        'batch': batch_trt,
    } if dataset_name == "bbbc021" else {
        'X': (img_ctrl, img_trt),
        'mols': mol,
        'y_id': y2id[y["trt"][idx_trt]],
        'file_names': (img_file_ctrl, img_file_trt),
        'idx_trt': idx_trt,
        'idx_ctrl': idx_ctrl,
        'batch': batch_trt,
    }

def read_files_batch(file_names, mols, mol2id, y2id, y, transform, image_path, dataset_name, idx):
    """
    Read and process batch images.

    Args:
        file_names (list): List of file names for the samples.
        mols (list): List of molecule information for the samples.
        mol2id (dict): Mapping from molecule names to IDs.
        y2id (dict): Mapping from annotation names to IDs.
        y (list): List of annotation information for the samples.
        transform (callable): Transformation to apply to the images.
        image_path (str): Path to the image folder.
        dataset_name (str): Name of the dataset.
        idx (int): Index of the sample to retrieve.

    Returns:
        dict: Dictionary containing processed image, molecule information, annotation ID, and file name.
    """
    img_file = file_names[idx]
    file_split = img_file.split('-')

    if dataset_name == "rxrx1":
        file_split = file_split[1].split("_")
        path = Path(image_path) / "_".join(file_split[:2]) / file_split[2]
        file = '_'.join(file_split[3:]) + ".npy"
    elif dataset_name in ["bbbc021", "bbbc025"]:
        file_split = file_split[0].split("_")
        path = Path(image_path) / file_split[0] / file_split[1]
        file = '_'.join(file_split[2:]) + ".npy"
    else:
        file_split = file_split[0].split("_")
        path = Path(image_path) / file_split[0] / f"{file_split[1]}_{file_split[2]}"
        file = '_'.join(file_split[1:]) + ".npy"

    img = np.load(path / file)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)  # Place channel dimension in front of the others
    img = transform(img)

    mol = mol2id[mols[idx]]

    return {
        'X': img,
        'mols': mol,
        'y_id': y2id[y[idx]],
        'file_names': img_file
    }

def convert_6ch_to_3ch(images):
    """
    Convert 6-channel images to 3-channel RGB composite images.

    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, 6, H, W), values in range [0, 1].

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 3, H, W), values in range [0, 1].
    """
    # Define the weights for each channel in RGB
    # Channel 1-6 mapped to specific colors
    weights = torch.tensor([
        [0, 0, 1],   # Channel 1 -> Blue
        [0, 1, 0],   # Channel 2 -> Green
        [1, 0, 0],   # Channel 3 -> Red
        [0, 0.5, 0.5],  # Channel 4 -> Cyan (lower intensity)
        [0.5, 0, 0.5],  # Channel 5 -> Magenta (lower intensity)
        [0.5, 0.5, 0],  # Channel 6 -> Yellow (lower intensity)
    ], dtype=images.dtype, device=images.device)

    # Perform matrix multiplication to combine channels
    # Shape transformation: (batch_size, 6, H, W) -> (batch_size, 3, H, W)
    images_rgb = torch.einsum('bchw,cn->bnhw', images, weights)

    # Clip the result to ensure it's within [0, 1]
    images_rgb = torch.clamp(images_rgb, -1, 1)

    return images_rgb

def convert_5ch_to_3ch(images):
    """
    Convert 5-channel images to 3-channel RGB composite images.

    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, 5, H, W), values in range [0, 1] or [-1, 1].

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 3, H, W), values in range [0, 1].
    """
    images_rgb = images[:, :3, :, :]
    return images_rgb
