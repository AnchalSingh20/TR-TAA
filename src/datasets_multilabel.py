"""
Multi-label PascalVOC classification dataset.

PascalVOCClassification (src/datasets.py) returns the first positive class
of each image (single-label). Following the TAA evaluation protocol
(Pulfer et al., CVPRW 2025), all classification results in the paper use
multi-label classification scored with mAP, provided by this module.

Usage:
    from src.datasets_multilabel import (
        PascalVOCClassificationMultiLabel,
        get_pascalvoc_classification_multilabel_loader,
    )

The returned label tensor is a multi-hot vector of shape (20,), where
labels[i] = 1.0 iff class i is present in the image.

Note: image-loading, resize, and tensor conversion code is byte-identical
to PascalVOCClassification — only the label tensor differs.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Re-export the class list so users can import from one place
from src.datasets import VOC_CLASSIFICATION_CLASSES


class PascalVOCClassificationMultiLabel(Dataset):
    """
    PascalVOC for MULTI-LABEL image classification.

    For each image, returns a multi-hot label vector of shape (20,) where
    labels[i] = 1.0 iff class i is annotated as present in the image.

    Args:
        root: Path to VOC2012 directory (e.g., /path/to/VOCdevkit/VOC2012)
        split: 'train' or 'val'
        image_size: Resize images to this size
        transform: Additional transforms (for robustness evaluation)
    """

    def __init__(self, root, split="val", image_size=518, transform=None):
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.split = split
        self.image_size = image_size
        self.additional_transform = transform

        self.images = []          # list of image file paths
        self.labels = []          # list of np.ndarray of shape (20,), float32
        self._load_classification_data()

    def _load_classification_data(self):
        """Load image IDs and their MULTI-LABEL class vectors from VOC files."""
        main_dir = os.path.join(self.root, "ImageSets", "Main")

        # image_id -> np.ndarray of shape (20,) with 0/1 per class
        image_label_map = {}

        num_classes = len(VOC_CLASSIFICATION_CLASSES)

        for cls_idx, cls_name in enumerate(VOC_CLASSIFICATION_CLASSES):
            txt_file = os.path.join(main_dir, f"{cls_name}_{self.split}.txt")
            if not os.path.exists(txt_file):
                continue

            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_id, flag = parts[0], int(parts[1])
                        # VOC convention:
                        #    1 -> positive (class present)
                        #    0 -> difficult (treated as positive in most setups)
                        #   -1 -> negative (class absent)
                        # We follow standard multi-label PascalVOC practice:
                        # only flag == 1 is positive.
                        if img_id not in image_label_map:
                            image_label_map[img_id] = np.zeros(
                                num_classes, dtype=np.float32
                            )
                        if flag == 1:
                            image_label_map[img_id][cls_idx] = 1.0

        # Filter out images with no positives (rare but possible)
        for img_id in sorted(image_label_map.keys()):
            lbl = image_label_map[img_id]
            if lbl.sum() == 0:
                continue
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(lbl)

        print(
            f"[PascalVOC Classification MULTI-LABEL] Loaded "
            f"{len(self.images)} images for split '{self.split}' "
            f"(avg labels/img = {np.mean([l.sum() for l in self.labels]):.2f})"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]                       # np.ndarray (20,)

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.additional_transform is not None:
            img = self.additional_transform(img)

        img_tensor = TF.to_tensor(img)                 # (3, H, W) in [0, 1]
        label_tensor = torch.from_numpy(label)         # (20,) float32

        return img_tensor, label_tensor, self.images[idx]


def get_pascalvoc_classification_multilabel_loader(
    root, split="val", batch_size=32, num_workers=4,
    image_size=518, transform=None,
):
    """DataLoader for multi-label PascalVOC classification."""
    dataset = PascalVOCClassificationMultiLabel(
        root, split, image_size, transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
