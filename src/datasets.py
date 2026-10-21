"""
Dataset utilities for PascalVOC and ImageNette.

These load datasets from LOCAL paths - no downloading.
Dataset roots are passed on the command line (e.g. --pascalvoc_root).

PascalVOC structure expected:
    VOCdevkit/VOC2012/
    ├── JPEGImages/
    ├── SegmentationClass/
    ├── ImageSets/
    │   ├── Main/          (for classification: train.txt, val.txt)
    │   └── Segmentation/  (for segmentation: train.txt, val.txt)
    └── ...

ImageNette structure expected:
    imagenette2/
    ├── train/
    │   ├── n01440764/
    │   ├── n02102040/
    │   └── ...
    └── val/
        ├── n01440764/
        └── ...
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# ============================================================
# PascalVOC Dataset
# ============================================================

# VOC class name to index mapping
VOC_CLASS_TO_IDX = {
    "background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3,
    "boat": 4, "bottle": 5, "bus": 6, "car": 7, "cat": 8,
    "chair": 9, "cow": 10, "diningtable": 11, "dog": 12,
    "horse": 13, "motorbike": 14, "person": 15, "pottedplant": 16,
    "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20,
}

# For classification: map VOC class names to 0-19 indices (no background)
VOC_CLASSIFICATION_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class PascalVOCClassification(Dataset):
    """
    PascalVOC for image classification.
    Uses the Main split (ImageSets/Main/*.txt).

    For each image, the class label is determined from the per-class
    annotation files. An image can belong to multiple classes;
    following standard practice, we use the first positive class.
    """

    def __init__(self, root, split="val", image_size=518, transform=None):
        """
        Args:
            root: Path to VOC2012 directory (e.g., /path/to/VOCdevkit/VOC2012)
            split: 'train' or 'val'
            image_size: Resize images to this size
            transform: Additional transforms (for robustness evaluation)
        """
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.split = split
        self.image_size = image_size
        self.additional_transform = transform

        # Build image list with labels from per-class annotation files
        self.images = []
        self.labels = []
        self._load_classification_data()

    def _load_classification_data(self):
        """Load image IDs and their class labels from VOC annotation files."""
        main_dir = os.path.join(self.root, "ImageSets", "Main")

        # Collect all image IDs that appear in this split for any class
        image_class_map = {}

        for cls_idx, cls_name in enumerate(VOC_CLASSIFICATION_CLASSES):
            txt_file = os.path.join(main_dir, f"{cls_name}_{self.split}.txt")
            if not os.path.exists(txt_file):
                continue

            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_id, flag = parts[0], int(parts[1])
                        if flag == 1:  # Positive sample for this class
                            if img_id not in image_class_map:
                                image_class_map[img_id] = cls_idx

        # Sort for reproducibility
        for img_id in sorted(image_class_map.keys()):
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(image_class_map[img_id])

        print(f"[PascalVOC Classification] Loaded {len(self.images)} images "
              f"for split '{self.split}'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        # Resize to fixed size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Apply additional transform (for robustness testing)
        if self.additional_transform is not None:
            img = self.additional_transform(img)

        # Convert to tensor in [0, 1]
        img_tensor = TF.to_tensor(img)

        return img_tensor, label, self.images[idx]


class PascalVOCSegmentation(Dataset):
    """
    PascalVOC for semantic segmentation.
    Uses the Segmentation split (ImageSets/Segmentation/*.txt).
    """

    def __init__(self, root, split="val", image_size=518, transform=None):
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")
        self.split = split
        self.image_size = image_size
        self.additional_transform = transform

        # Load image IDs from segmentation split
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        # Filter to only existing files
        valid_ids = []
        for img_id in self.image_ids:
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_ids.append(img_id)
        self.image_ids = valid_ids

        print(f"[PascalVOC Segmentation] Loaded {len(self.image_ids)} images "
              f"for split '{self.split}'")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # Palette mode, class indices

        # Resize image
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        # Resize mask with nearest neighbor to preserve class labels
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Apply additional transform (for robustness testing)
        # The transform object must have a boolean attribute `is_geometric`
        # to indicate whether the mask should also be transformed.
        if self.additional_transform is not None:
            is_geometric = getattr(self.additional_transform, 'is_geometric', False)
            img = self.additional_transform(img)
            if is_geometric:
                # Apply the same geometric transform to the mask
                # Convert mask to numpy first to preserve palette indices
                mask_np = np.array(mask)
                mask_pil = Image.fromarray(mask_np)
                mask_pil = self.additional_transform(mask_pil)
                mask = mask_pil

        # Convert to tensors
        img_tensor = TF.to_tensor(img)  # (3, H, W) in [0, 1]
        mask_tensor = torch.from_numpy(np.array(mask)).long()  # (H, W)

        # VOC uses 255 for boundary/ignore regions
        mask_tensor[mask_tensor == 255] = -1  # Will be ignored in loss/metric

        return img_tensor, mask_tensor, img_path


# ============================================================
# ImageNette Dataset
# ============================================================

# ImageNette uses 10 ImageNet class folders
IMAGENETTE_WNID_TO_IDX = {
    "n01440764": 0,  # tench
    "n02102040": 1,  # English springer
    "n02979186": 2,  # cassette player
    "n03000684": 3,  # chain saw
    "n03028079": 4,  # church
    "n03394916": 5,  # French horn
    "n03417042": 6,  # garbage truck
    "n03425413": 7,  # gas pump
    "n03445777": 8,  # golf ball
    "n03888257": 9,  # parachute
}


class ImageNetteDataset(Dataset):
    """ImageNette dataset loaded from local directory."""

    def __init__(self, root, split="val", image_size=518):
        self.root = os.path.join(root, split)
        self.image_size = image_size
        self.images = []
        self.labels = []

        for wnid, label_idx in IMAGENETTE_WNID_TO_IDX.items():
            class_dir = os.path.join(self.root, wnid)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".JPEG")):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

        print(f"[ImageNette] Loaded {len(self.images)} images for split '{split}'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        return img_tensor, label, self.images[idx]


# ============================================================
# Helper: Create dataloaders
# ============================================================

def get_pascalvoc_classification_loader(root, split="val", batch_size=32,
                                         num_workers=4, image_size=518,
                                         transform=None):
    dataset = PascalVOCClassification(root, split, image_size, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def get_pascalvoc_segmentation_loader(root, split="val", batch_size=16,
                                       num_workers=4, image_size=518,
                                       transform=None):
    dataset = PascalVOCSegmentation(root, split, image_size, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def get_imagenette_loader(root, split="val", batch_size=32,
                           num_workers=4, image_size=518):
    dataset = ImageNetteDataset(root, split, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)
