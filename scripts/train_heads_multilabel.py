"""
Train the multi-label classification head (linear layer on the CLS token,
BCEWithLogitsLoss, sigmoid outputs, mAP evaluation) on frozen DINOv2
features, PASCAL VOC 2012 train split. Used for all classification results.

Saves {output_dir}/{model_name}_classification_head_multilabel.pth

USAGE
    python scripts/train_heads_multilabel.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14 --output_dir ./checkpoints
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_dinov2_model, get_model_info, normalize_for_model
from src.heads_multilabel import MultiLabelClassificationHead
from src.datasets_multilabel import (
    get_pascalvoc_classification_multilabel_loader,
)
from src.metrics_multilabel import evaluate_classification_multilabel


def train_multilabel_head(args):
    device = args.device
    model_info = get_model_info(args.model_name)
    feature_dim = model_info["dim"]

    print(f"\n{'='*60}")
    print(f"Training MULTI-LABEL CLASSIFICATION head for {args.model_name}")
    print(f"Feature dim: {feature_dim}, Num classes: 20")
    print(f"Loss: BCEWithLogitsLoss   |   Metric: mAP")
    print(f"{'='*60}\n")

    # Load frozen backbone
    backbone = get_dinov2_model(args.model_name, device)

    # Multi-label linear head
    head = MultiLabelClassificationHead(
        feature_dim=feature_dim, num_classes=20
    ).to(device)

    # Dataloaders
    train_loader = get_pascalvoc_classification_multilabel_loader(
        args.pascalvoc_root, split="train",
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size,
    )
    val_loader = get_pascalvoc_classification_multilabel_loader(
        args.pascalvoc_root, split="val",
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size,
    )

    optimizer = optim.Adam(head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.BCEWithLogitsLoss()

    best_map = 0.0

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels, _ in tqdm(train_loader,
                                      desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)               # (B, 20) float

            with torch.no_grad():
                images_norm = normalize_for_model(images)
                features = backbone.forward_features(images_norm)
                cls_token = features["x_norm_clstoken"]

            logits = head(cls_token)                  # (B, 20)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss   += loss.item() * images.shape[0]
            total_samples += images.shape[0]

        scheduler.step()

        train_loss = total_loss / max(total_samples, 1)
        val_map = evaluate_classification_multilabel(
            backbone, head, val_loader, device, desc="Validation"
        )

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}  Val mAP={val_map:.2f}%")

        if val_map > best_map:
            best_map = val_map
            save_path = os.path.join(
                args.output_dir,
                f"{args.model_name}_classification_head_multilabel.pth"
            )
            torch.save(head.state_dict(), save_path)
            print(f"  -> Saved best multi-label head (Val mAP: {best_map:.2f}%)")

    print(f"\nBest validation mAP: {best_map:.2f}%")
    return head


def main():
    parser = argparse.ArgumentParser(description="Train multi-label classification head")
    parser.add_argument("--pascalvoc_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_multilabel_head(args)


if __name__ == "__main__":
    main()
