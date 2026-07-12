#!/usr/bin/env python3
"""Replace cellpose-masked single-cell crops with un-masked crops from original field images.

Current: single_cell_images/0h/Dark/57926_22.png  ← cellpose crop (background zeroed)
Target:  same path, but pixel content from field_images/0h/Dark/images/57926.jpg,
         cropped to the cell's bounding-box + centered on 128x128 canvas,
         preserving the original bright-field background and membrane gradient.

Principal: mask is used ONLY to locate bounding-box; pixel values come from field image.
"""
from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

FIELD_DIR = Path("data/raw/microalgae_v1/field_images")
CROP_DIR = Path("data/raw/microalgae_v1/single_cell_images")
META = Path("data/processed/microalgae_v1/views/field") / "metadata.csv"
CANVAS = 128
CROP_RE = re.compile(r"^(\d+)_(\d+)\.png$")


def build_map():
    """field_id -> {image_path, mask_path, time_light_dir}."""
    import pandas as pd
    df = pd.read_csv(META)
    m = {}
    for _, r in df.iterrows():
        fid = str(int(r["field_id"]))
        # image_relpath e.g. "0h/Dark/images/57926.jpg"
        parts = Path(r["image_relpath"]).parts
        time_light = Path(parts[0]) / parts[1]      # 0h/Dark
        m[fid] = {
            "image": FIELD_DIR / r["image_relpath"],
            "mask":  FIELD_DIR / r["mask_relpath"],
            "dir":   time_light,
        }
    return m


def extract_bbox(mask_arr, instance_id):
    """Row/col bounds of the instance label in the full mask image."""
    ys, xs = np.where(mask_arr == instance_id)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def make_crop(field_img, bbox, mask_arr=None, canvas=CANVAS):
    """Centered 128×128 crop from the RAW field image, anchored on the cell's mask bbox.
    The crop always comes from the same field image, preserving natural texture continuity.
    Pads only when the cell is within 64 px of the field edge, using adjacent field pixels."""
    y0, y1, x0, x1 = bbox
    h, w = field_img.shape[:2]
    # Anchor on bbox centroid
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    # 128×128 centred window
    sy0, sx0 = cy - canvas // 2, cx - canvas // 2
    sy1, sx1 = sy0 + canvas, sx0 + canvas
    # clamp
    osy0, osx0 = sy0, sx0
    sy0_clip = max(0, sy0); sx0_clip = max(0, sx0)
    sy1_clip = min(h, sy1); sx1_clip = min(w, sx1)
    region = field_img[sy0_clip:sy1_clip, sx0_clip:sx1_clip].copy()
    # pad if cropped window extends beyond field edges
    out = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    dy = sy0_clip - osy0; dx = sx0_clip - osx0
    rh, rw = region.shape[:2]
    out[dy:dy+rh, dx:dx+rw] = region
    # edge-fill any remaining zero rows/cols by reflecting the last valid edge
    # (very rare; only near field borders where pad >0)
    if np.any(out == 0):
        for c in range(3):
            ch = out[:,:,c]
            # fill top/bottom
            for r in range(canvas):
                row = ch[r]
                nz = np.where(row > 0)[0]
                if len(nz) == 0: continue
                row[:nz[0]] = row[nz[0]]
                row[nz[-1]+1:] = row[nz[-1]]
    return Image.fromarray(out)


def main():
    bm = build_map()
    print(f"Loaded {len(bm)} field_id entries from metadata", flush=True)
    total, done, skipped = 0, 0, 0
    # cache: field_id -> np.array of its image (loaded once)
    img_cache = {}

    for crop_path in sorted(CROP_DIR.rglob("*.png")):
        m = CROP_RE.match(crop_path.name)
        if not m:
            continue
        field_id, instance = m.group(1), int(m.group(2))
        if field_id not in bm:
            skipped += 1
            continue
        rec = bm[field_id]
        total += 1
        mask_arr = np.asarray(Image.open(rec["mask"]).convert("L"))
        bbox = extract_bbox(mask_arr, instance)
        if bbox is None:
            skipped += 1
            continue
        # Load field image once per field_id
        if field_id not in img_cache:
            img_cache[field_id] = np.asarray(
                Image.open(str(rec["image"])).convert("RGB")
            )
        new = make_crop(img_cache[field_id], bbox, mask_arr)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        new.save(crop_path)
        done += 1
        if done % 1000 == 0:
            print(f"  {done}/{total} done | {skipped} skipped", flush=True)

    print(f"Total matched: {total}  written: {done}  skipped: {skipped}", flush=True)
    if done == 0 and skipped > 0:
        print("!! 0 files processed. Sample field_ids in map:", list(bm.keys())[:5])
        print("!! Sample crop files:")
        for p in sorted(CROP_DIR.rglob("*.png"))[:5]:
            print("   ", p)


if __name__ == "__main__":
    main()
