"""
Train the linear depth head on frozen DINOv2 features (NYU Depth v2).

Architecture: src.heads_depth.LinearDepthHead (1x1 conv on patch tokens,
predicting log-depth).
Loss: src.losses_depth.SiLogLoss (scale-invariant log loss).
Data: BTS-format NYUv2, 5000 randomly subsampled training pairs (fixed
seed); evaluation on the 654-image Eigen test split.

Saves {output_dir}/{model_name}_depth_head_linear.pth (+ _history.pt)

USAGE
    python scripts/train_heads_depth.py --nyuv2_root <NYUv2> \
        --model_name dinov2_vitb14 --output_dir ./checkpoints \
        --epochs 30 --batch_size 8 --lr 1e-3 --subsample 5000 --seed 42
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_dinov2_model, get_model_info, normalize_for_model
from src.heads_depth import LinearDepthHead, log_depth_to_meters
from src.losses_depth import SiLogLoss
from src.metrics_depth import compute_depth_metrics, average_metrics
from src.datasets_nyuv2 import NYUDepthV2BTS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(backbone, head, val_loader, device, desc="Validation",
             min_depth=0.001, max_depth=10.0):
    """
    Run validation on the full test set. Returns averaged metrics dict.
    """
    backbone.eval()
    head.eval()

    per_image_metrics = []

    for batch in tqdm(val_loader, desc=desc, leave=False):
        images, depths, valid_masks, _ = batch
        images = images.to(device)
        # depths and valid_masks stay on CPU - we compute metrics per image

        x = normalize_for_model(images)
        feats = backbone.forward_features(x)
        patch = feats["x_norm_patchtokens"]            # (B, N, D)

        log_pred = head(patch)                         # (B, H_out, W_out)
        pred_meters = log_depth_to_meters(log_pred,
                                          min_depth=min_depth,
                                          max_depth=max_depth)

        pred_cpu = pred_meters.cpu()

        # compute per-image metrics; align prediction to GT 480x640
        # (compute_depth_metrics handles alignment internally)
        B = pred_cpu.shape[0]
        for i in range(B):
            m = compute_depth_metrics(
                pred_cpu[i], depths[i], valid_masks[i],
                min_depth=min_depth, max_depth=max_depth,
            )
            per_image_metrics.append(m)

    return average_metrics(per_image_metrics)


def train_depth_head(args):
    set_seed(args.seed)
    device = args.device
    model_info = get_model_info(args.model_name)
    feature_dim = model_info["dim"]
    patch_size = model_info["patch_size"]

    print(f"\n{'=' * 60}")
    print(f"Training LINEAR DEPTH HEAD for {args.model_name}")
    print(f"Feature dim: {feature_dim}, patch size: {patch_size}")
    print(f"Input image size: {args.image_size}")
    print(f"Loss: SiLogLoss (lambda={args.silog_lambda})")
    print(f"Train subsample: {args.subsample} / 50688")
    print(f"{'=' * 60}\n")

    # ---- Backbone (frozen) ----
    backbone = get_dinov2_model(args.model_name, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ---- Head (the only trainable thing) ----
    head = LinearDepthHead(
        feature_dim=feature_dim,
        patch_size=patch_size,
        image_size=args.image_size,
        output_size=(480, 640),
    ).to(device)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head parameter count: {n_params}")

    # ---- Datasets ----
    print("Loading datasets...")
    full_train = NYUDepthV2BTS(
        args.nyuv2_root, split="train", image_size=args.image_size,
    )
    val_set = NYUDepthV2BTS(
        args.nyuv2_root, split="test", image_size=args.image_size,
    )

    # Random subsample (fixed seed so the choice is reproducible)
    n_full = len(full_train)
    n_sub = min(args.subsample, n_full)
    rng = np.random.RandomState(args.seed)
    subset_indices = rng.choice(n_full, size=n_sub, replace=False)
    train_set = Subset(full_train, subset_indices.tolist())
    print(f"  train: {len(train_set)} (subsampled from {n_full})")
    print(f"  val:   {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Loss, optimizer, scheduler ----
    criterion = SiLogLoss(lambda_var=args.silog_lambda, scale=10.0,
                          min_depth=args.min_depth)
    optimizer = optim.Adam(head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Train loop ----
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir, f"{args.model_name}_depth_head_linear.pth"
    )

    best_rmse = float("inf")
    history = []

    for epoch in range(args.epochs):
        # ---- Train phase ----
        head.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")
        for images, depths, valid_masks, _ in pbar:
            images      = images.to(device, non_blocking=True)
            depths      = depths.to(device, non_blocking=True)
            valid_masks = valid_masks.to(device, non_blocking=True)

            with torch.no_grad():
                x = normalize_for_model(images)
                feats = backbone.forward_features(x)
                patch = feats["x_norm_patchtokens"]            # (B, N, D)

            log_pred = head(patch)                              # (B, H, W)
            loss = criterion(log_pred, depths, valid_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            total_count += images.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = total_loss / max(total_count, 1)

        # ---- Validation phase ----
        val_metrics = evaluate(backbone, head, val_loader, device,
                               desc=f"Epoch {epoch+1} [val]",
                               min_depth=args.min_depth,
                               max_depth=args.max_depth)

        rmse = val_metrics["RMSE"]
        abs_rel = val_metrics["AbsRel"]
        d1 = val_metrics["delta1"]
        print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}  "
              f"Val RMSE={rmse:.4f}  AbsRel={abs_rel:.4f}  d1={d1:.4f}")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            **val_metrics,
        })

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "head_state_dict": head.state_dict(),
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "config": vars(args),
            }, save_path)
            print(f"  -> Saved best head (Val RMSE: {best_rmse:.4f})")

    # Save history alongside the checkpoint
    history_path = save_path.replace(".pth", "_history.pt")
    torch.save(history, history_path)
    print(f"\nBest Val RMSE: {best_rmse:.4f}")
    print(f"Saved:")
    print(f"  {save_path}")
    print(f"  {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train linear depth head")
    parser.add_argument("--nyuv2_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=5000)
    parser.add_argument("--silog_lambda", type=float, default=0.85)
    parser.add_argument("--min_depth", type=float, default=0.001)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_depth_head(args)


if __name__ == "__main__":
    main()
