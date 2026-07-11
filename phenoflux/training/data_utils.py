import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from PIL import Image


def _load_microalgae_rgb(image_path, sample_key, image_size=128):
    """Load an RGB microalgae image to [-1, 1].

    Single-cell Cellpose crops are scale-bearing images, so PNG crops are
    centered on a fixed canvas instead of stretched to fill it. Whole-field JPGs
    are still resized because they are full field-of-view images.
    """
    path = Path(image_path) / sample_key
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    if str(sample_key).lower().endswith(".png"):
        img = _pad_or_shrink_to_canvas(img, image_size=image_size)
        return img / 127.5 - 1.0
    if img.shape[-2:] != (image_size, image_size):
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return img / 127.5 - 1.0


def _pad_or_shrink_to_canvas(img, image_size=128):
    """Keep crop pixel scale when possible; only shrink rare oversized crops."""
    _, h, w = img.shape
    if h > image_size or w > image_size:
        scale = image_size / max(h, w)
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        _, h, w = img.shape

    canvas = torch.zeros((img.shape[0], image_size, image_size), dtype=img.dtype, device=img.device)
    y0 = (image_size - h) // 2
    x0 = (image_size - w) // 2
    canvas[:, y0 : y0 + h, x0 : x0 + w] = img
    return canvas


def _load_microalgae_png(image_path, sample_key, image_size=128):
    """Backward-compatible wrapper for crop PNG loading."""
    return _load_microalgae_rgb(image_path, sample_key, image_size=image_size)


def centered_noise(shape, sigma, device="cpu"):
    """Generate noise with a Gaussian spatial envelope centered in the image.

    Args:
        shape: (B, C, H, W) output shape.
        sigma: envelope width in normalized [-1,1] coordinates.
               sigma=0 → uniform noise; sigma=0.3–0.5 → single centered cell.
        device: torch device.

    Returns:
        torch.Tensor of shape ``shape`` in [-1, 1].
    """
    if sigma <= 0:
        return torch.randn(shape, device=device)

    _, _, H, W = shape
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r2 = yy**2 + xx**2
    envelope = torch.exp(-r2 / (2 * sigma**2))  # 1 at centre, →0 at corners
    noise = torch.randn(shape, device=device)
    return noise * envelope.unsqueeze(0).unsqueeze(0)


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

def read_files_pert(file_names, mols, mol2id, y2id, dose, y, transform, image_path, dataset_name, idx, multimodal, batch, iter_ctrl, pairing_mode='batch_random', precomputed_pairing=None, cluster_map=None, augment_strength='default'):
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

        # --- Pairing strategy selection (ADR-002 data-quality improvement) ---
        if pairing_mode == 'merfish_nn' and precomputed_pairing is not None:
            # Strategy B: precomputed MERFISH nearest-neighbor pairing
            precomputed_ctrl = precomputed_pairing.get(img_file_trt)
            if precomputed_ctrl is not None:
                # Find index of precomputed control cell in file_names["ctrl"]
                ctrl_list = list(file_names["ctrl"])
                try:
                    idx_ctrl = ctrl_list.index(precomputed_ctrl)
                    img_file_ctrl = file_names["ctrl"][idx_ctrl]
                except ValueError:
                    # Fall back to batch_random if precomputed ctrl not in fold
                    ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
                    if len(ctrl_indices_same_batch) == 0:
                        raise ValueError(f"No control samples found in batch {batch_trt}")
                    idx_ctrl = np.random.choice(ctrl_indices_same_batch)
                    img_file_ctrl = file_names["ctrl"][idx_ctrl]
            else:
                # Treated cell not in precomputed index -> fall back
                ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
                if len(ctrl_indices_same_batch) == 0:
                    raise ValueError(f"No control samples found in batch {batch_trt}")
                idx_ctrl = np.random.choice(ctrl_indices_same_batch)
                img_file_ctrl = file_names["ctrl"][idx_ctrl]

        elif pairing_mode == 'cluster_match' and cluster_map is not None:
            # Strategy C: same cluster_type pairing
            trt_cluster = cluster_map.get(img_file_trt)
            if trt_cluster is not None:
                ctrl_same_cluster = np.where(
                    np.array([cluster_map.get(fn) == trt_cluster for fn in file_names["ctrl"]])
                )[0]
                if len(ctrl_same_cluster) > 0:
                    idx_ctrl = np.random.choice(ctrl_same_cluster)
                    img_file_ctrl = file_names["ctrl"][idx_ctrl]
                else:
                    # Fall back to batch_random
                    ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
                    idx_ctrl = np.random.choice(ctrl_indices_same_batch)
                    img_file_ctrl = file_names["ctrl"][idx_ctrl]
            else:
                ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
                idx_ctrl = np.random.choice(ctrl_indices_same_batch)
                img_file_ctrl = file_names["ctrl"][idx_ctrl]

        else:
            # Strategy A (default) or fallback: random same-batch pairing
            ctrl_indices_same_batch = np.where(batch["ctrl"] == batch_trt)[0]
            if len(ctrl_indices_same_batch) == 0:
                raise ValueError(f"No control samples found in the same batch as the treated sample (batch: {batch_trt}).")

            idx_ctrl = np.random.choice(ctrl_indices_same_batch)
            img_file_ctrl = file_names["ctrl"][idx_ctrl]

    if dataset_name in {"microalgae", "microalgae_field"}:
        # Direct image load by filename; same-batch ctrl pairing already done above.
        img_ctrl = _load_microalgae_rgb(image_path, img_file_ctrl)
        img_trt = _load_microalgae_rgb(image_path, img_file_trt)

        # Range-safe augmentation for RGB data already normalized to [-1,1].
        # Apply identical transforms to ctrl+trt so the learned mapping is not
        # confounded by augmentation mismatch.
        if getattr(transform, "augment", False):
            _astrength = augment_strength

            # --- Always-on: random flips (safe for [-1,1]) ---
            if torch.rand(1).item() < 0.3:
                img_ctrl, img_trt = torch.flip(img_ctrl, [-1]), torch.flip(img_trt, [-1])
            if torch.rand(1).item() < 0.3:
                img_ctrl, img_trt = torch.flip(img_ctrl, [-2]), torch.flip(img_trt, [-2])

            # --- Strong augmentation: spatial jitter + intensity scaling ---
            if _astrength == 'strong':
                # Random spatial shift (2-8 pixels), zero-padded
                shift_y = np.random.randint(-8, 9)
                shift_x = np.random.randint(-8, 9)
                if shift_y != 0 or shift_x != 0:
                    img_ctrl = torch.roll(img_ctrl, shifts=(shift_y, shift_x), dims=(-2, -1))
                    img_trt = torch.roll(img_trt, shifts=(shift_y, shift_x), dims=(-2, -1))

                # Random per-channel intensity scaling (0.9-1.1)
                intensity_scale = 0.9 + 0.2 * torch.rand(1).item()
                img_ctrl = img_ctrl * intensity_scale
                img_trt = img_trt * intensity_scale

                # Add small Gaussian noise (std=0.02, on [-1,1] scale)
                noise_std = 0.02
                img_ctrl = img_ctrl + torch.randn_like(img_ctrl) * noise_std
                img_trt = img_trt + torch.randn_like(img_trt) * noise_std
                # Clamp back to [-1,1]
                img_ctrl = torch.clamp(img_ctrl, -1.0, 1.0)
                img_trt = torch.clamp(img_trt, -1.0, 1.0)

        return {
            'X': (img_ctrl, img_trt),
            'mols': mol2id[mols["trt"][idx_trt]],
            'y_id': y2id[y["trt"][idx_trt]],
            'file_names': (img_file_ctrl, img_file_trt),
            'idx_trt': idx_trt,
            'idx_ctrl': idx_ctrl,
            'batch': batch_trt,
        }

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
