"""
Train the downstream heads (single-label classification and semantic
segmentation) on frozen DINOv2 features, PASCAL VOC 2012 train split.

The segmentation head is used for all segmentation results. The
single-label classification head is not used for reported results (see
train_heads_multilabel.py).

USAGE
    python scripts/train_heads.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14 --task both --output_dir ./checkpoints
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_dinov2_model, get_model_info, normalize_for_model
from src.heads import LinearClassificationHead, SegmentationHead
from src.datasets import (
    PascalVOCClassification, PascalVOCSegmentation,
    get_pascalvoc_classification_loader, get_pascalvoc_segmentation_loader,
)
from src.metrics import evaluate_classification, evaluate_segmentation


def train_classification_head(args):
    """Train a linear classification head on PascalVOC."""
    device = args.device
    model_info = get_model_info(args.model_name)
    feature_dim = model_info["dim"]

    print(f"\n{'='*60}")
    print(f"Training CLASSIFICATION head for {args.model_name}")
    print(f"Feature dim: {feature_dim}, Num classes: 20")
    print(f"{'='*60}\n")

    # Load backbone
    backbone = get_dinov2_model(args.model_name, device)

    # Create head
    head = LinearClassificationHead(feature_dim, num_classes=20).to(device)

    # Dataloaders
    train_loader = get_pascalvoc_classification_loader(
        args.pascalvoc_root, split="train", batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
    )
    val_loader = get_pascalvoc_classification_loader(
        args.pascalvoc_root, split="val", batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
    )

    # Training setup
    optimizer = optim.Adam(head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            # Extract features (no gradient through backbone)
            with torch.no_grad():
                images_norm = normalize_for_model(images)
                features = backbone.forward_features(images_norm)
                cls_token = features["x_norm_clstoken"]

            # Forward through head
            logits = head(cls_token)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total_samples += images.shape[0]

        scheduler.step()

        train_acc = 100.0 * total_correct / total_samples
        train_loss = total_loss / total_samples

        # Evaluate on validation set
        val_acc = evaluate_classification(backbone, head, val_loader, device,
                                           desc="Validation")

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(
                args.output_dir,
                f"{args.model_name}_classification_head.pth"
            )
            torch.save(head.state_dict(), save_path)
            print(f"  -> Saved best model (Val Acc: {best_acc:.1f}%)")

    print(f"\nBest validation accuracy: {best_acc:.1f}%")
    return head


def train_segmentation_head(args):
    """Train a segmentation head on PascalVOC."""
    device = args.device
    model_info = get_model_info(args.model_name)
    feature_dim = model_info["dim"]
    patch_size = model_info["patch_size"]

    print(f"\n{'='*60}")
    print(f"Training SEGMENTATION head for {args.model_name}")
    print(f"Feature dim: {feature_dim}, Patch size: {patch_size}, Num classes: 21")
    print(f"{'='*60}\n")

    # Load backbone
    backbone = get_dinov2_model(args.model_name, device)

    # Create head
    head = SegmentationHead(
        feature_dim=feature_dim,
        num_classes=21,
        patch_size=patch_size,
        image_size=args.image_size,
    ).to(device)

    # Dataloaders
    train_loader = get_pascalvoc_segmentation_loader(
        args.pascalvoc_root, split="train", batch_size=args.seg_batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
    )
    val_loader = get_pascalvoc_segmentation_loader(
        args.pascalvoc_root, split="val", batch_size=args.seg_batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
    )

    # Training setup
    optimizer = optim.Adam(head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_miou = 0.0

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0
        total_pixels = 0

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            # Extract features (no gradient through backbone)
            with torch.no_grad():
                images_norm = normalize_for_model(images)
                features = backbone.forward_features(images_norm)
                cls_token = features["x_norm_clstoken"]
                patch_tokens = features["x_norm_patchtokens"]

            # Forward through head
            logits = head(cls_token, patch_tokens)

            # Resize logits to match mask size if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            valid_pixels = (masks != -1).sum().item()
            total_loss += loss.item() * valid_pixels
            total_pixels += valid_pixels

        scheduler.step()

        train_loss = total_loss / max(total_pixels, 1)

        # Evaluate on validation set
        val_miou = evaluate_segmentation(backbone, head, val_loader, device,
                                          num_classes=21, desc="Validation")

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val mIoU={val_miou:.1f}%")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_path = os.path.join(
                args.output_dir,
                f"{args.model_name}_segmentation_head.pth"
            )
            torch.save(head.state_dict(), save_path)
            print(f"  -> Saved best model (Val mIoU: {best_miou:.1f}%)")

    print(f"\nBest validation mIoU: {best_miou:.1f}%")
    return head


def main():
    parser = argparse.ArgumentParser(description="Train downstream task heads")
    parser.add_argument("--pascalvoc_root", type=str, required=True,
                        help="Path to VOCdevkit/VOC2012")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--task", type=str, default="both",
                        choices=["classification", "segmentation", "both"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seg_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task in ["classification", "both"]:
        train_classification_head(args)

    if args.task in ["segmentation", "both"]:
        train_segmentation_head(args)


if __name__ == "__main__":
    main()
