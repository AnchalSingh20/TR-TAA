"""
NYU Depth v2 dataset loader for the BTS-format PNG-pair release.

This is a DIFFERENT loader from the labeled .mat-based one. This release
of NYUv2 (re-distributed by the BTS authors and now the de-facto standard
in modern depth-prediction literature) provides:

  - 654 test pairs in a flat directory (matches the official Eigen test split)
  - ~50K train pairs organized by scene (BTS-augmented, larger than the
    original Eigen 795)

Both are referenced through two CSV files that list (rgb_path, depth_path)
pairs relative to the dataset root.

Expected layout (BTS-format release):
    <root>/
        nyu_data/
            data/
                nyu2_test/                   654 pairs (flat dir)
                    00000_colors.png         RGB,  480x640, uint8
                    00000_depth.png          depth, 480x640, uint16 (mm)
                    00001_colors.png
                    00001_depth.png
                    ...
                nyu2_train/
                    <scene_name>_out/        scene-organized
                        37.jpg               RGB,  480x640, uint8
                        37.png               depth, 480x640, uint16 (mm)
                        ...
                nyu2_test.csv                manifest: rgb_path,depth_path
                nyu2_train.csv               manifest: rgb_path,depth_path

Depth conventions:
  - uint16 PNG values are DEPTH IN MILLIMETERS
  - Divide by 1000.0 for meters
  - Depth = 0 means INVALID PIXEL (sensor returned nothing)
  - Most pixels are valid in this filled-in version (the .mat-based
    Levin in-painting was applied before redistributing)

The dataset returns:
  - image tensor (3, image_size, image_size) in [0,1], resized square
  - depth tensor at NATIVE 480x640 resolution, in METERS, with 0=invalid
  - valid_mask tensor (480, 640) bool, True where depth > 0
  - relative path string (for traceability)
"""

import os
import csv
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# Native NYUv2 resolution (kept for ground-truth depth)
NYUV2_H = 480
NYUV2_W = 640

# Depth scale: uint16 PNG values are millimeters
DEPTH_SCALE_MM = 1000.0

# Expected file counts for this BTS-format release
EXPECTED_TEST_PAIRS = 654


def _read_manifest_csv(csv_path):
    """
    Read a BTS-format manifest CSV. Each row is:
        <rgb_path>,<depth_path>
    where paths are relative to the dataset root (e.g.
    'data/nyu2_test/00000_colors.png').

    Returns:
        list of (rgb_rel_path, depth_rel_path) tuples
    """
    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            rgb_path, depth_path = row[0].strip(), row[1].strip()
            if rgb_path and depth_path:
                pairs.append((rgb_path, depth_path))
    return pairs


def _load_depth_meters(path):
    """
    Load a BTS-format depth PNG and convert to meters.

    Returns:
        np.ndarray (H, W) float32 depth in meters; 0 means invalid pixel
    """
    arr = np.array(Image.open(path))
    if arr.dtype != np.uint16:
        # Some redistributions store as 8-bit; convert if needed.
        # We don't expect this for the BTS release but handle it gracefully.
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) * (10.0 / 255.0)   # scale to meters
            return arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)
    return arr.astype(np.float32) / DEPTH_SCALE_MM


class NYUDepthV2BTS(Dataset):
    """
    PyTorch Dataset for NYU Depth v2 (BTS-format PNG-pair release).

    Args:
        root: directory containing nyu_data/data/  (the BTS layout)
              e.g. ./data/nyuv2
        split: 'train' or 'test'
        image_size: target square size for RGB (default 518 for DINOv2/14)
        transform: optional torchvision-style transform applied to the
                   RGB PIL image AFTER resize, BEFORE to_tensor()

    __getitem__ returns:
        image: (3, image_size, image_size) float tensor in [0, 1]
        depth: (480, 640) float tensor, in METERS (native resolution)
        valid_mask: (480, 640) bool tensor, True where depth > 0
        rel_path: relative RGB path (for traceability)
    """

    def __init__(self, root, split="test", image_size=518, transform=None):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform

        if split == "test":
            csv_path = os.path.join(root, "nyu_data", "data", "nyu2_test.csv")
        elif split == "train":
            csv_path = os.path.join(root, "nyu_data", "data", "nyu2_train.csv")
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Manifest CSV not found at {csv_path}. "
                f"Expected layout: {root}/nyu_data/data/nyu2_{split}.csv"
            )

        # CSV paths are relative to <root>/nyu_data/, not <root>.
        # Sample row: 'data/nyu2_test/00000_colors.png,data/nyu2_test/00000_depth.png'
        self.dataset_base = os.path.join(root, "nyu_data")

        self.pairs = _read_manifest_csv(csv_path)

        # Filter out any pair whose files don't exist on disk (defensive)
        kept = []
        missing = 0
        for rgb_rel, depth_rel in self.pairs:
            rgb_abs = os.path.join(self.dataset_base, rgb_rel)
            depth_abs = os.path.join(self.dataset_base, depth_rel)
            if os.path.exists(rgb_abs) and os.path.exists(depth_abs):
                kept.append((rgb_rel, depth_rel))
            else:
                missing += 1
        if missing > 0:
            print(f"  Warning: {missing}/{len(self.pairs)} CSV entries "
                  f"missing on disk; skipping them.")
        self.pairs = kept

        print(f"[NYUv2 BTS] split={split}: {len(self.pairs)} pairs loaded "
              f"from {csv_path}")
        if split == "test" and len(self.pairs) != EXPECTED_TEST_PAIRS:
            print(f"  Note: expected {EXPECTED_TEST_PAIRS} test pairs "
                  f"(canonical Eigen test split), got {len(self.pairs)}.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_rel, depth_rel = self.pairs[idx]
        rgb_path   = os.path.join(self.dataset_base, rgb_rel)
        depth_path = os.path.join(self.dataset_base, depth_rel)

        # RGB
        img_pil = Image.open(rgb_path).convert("RGB")
        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.transform is not None:
            img_pil = self.transform(img_pil)
        img_tensor = TF.to_tensor(img_pil)              # (3, H, W) in [0,1]

        # Depth: keep at native 480x640
        depth_np = _load_depth_meters(depth_path)
        depth_tensor = torch.from_numpy(depth_np)
        valid_mask = depth_tensor > 0.0

        return img_tensor, depth_tensor, valid_mask, rgb_rel
