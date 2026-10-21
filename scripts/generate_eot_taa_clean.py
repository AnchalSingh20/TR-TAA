"""
Generate TR-TAA adversarial images on PASCAL VOC 2012 val (classification
and segmentation sets).

Uses the same mean vector mu, image loading and preprocessing as the TAA
baseline (generate_taa_baseline.py), so the two attacks are evaluated on
identical inputs. The fixed clean reference is used (Algorithm 1).

USAGE (run once per K)
    python scripts/generate_eot_taa_clean.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14 --K 8
    python scripts/generate_eot_taa_clean.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14 --K 4

    # seed-stability runs (ViT-B/14, seeds 0/1/2) -> cached_images_clean_seed<s>/
    python scripts/generate_eot_taa_clean.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vitb14 --K 8 --seed 0

OUTPUTS (outputs/cached_images_clean/<model_name>/, or
         outputs/cached_images_clean_seed<s>/<model_name>/ with --seed)
    cls_adv_images_K{K}.pt     adversarial images, classification set
    cls_clean_images.pt        clean classification images
    cls_labels.pt              labels
    seg_adv_images_K{K}.pt     adversarial images, segmentation set
    seg_clean_images.pt        clean segmentation images
    seg_masks.pt               segmentation ground truth
    psnr_info_K{K}.pt          PSNR statistics
    mu_patch.pt                mean feature vector used for centering
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_dinov2_model, get_model_info
from src.heads import LinearClassificationHead, SegmentationHead
from src.datasets import (
    PascalVOCClassification,
    PascalVOCSegmentation,
    get_pascalvoc_classification_loader,
)
from src.attacks import compute_mean_features, compute_psnr
from src.eot_attack import eot_taa_attack


def main():
    parser = argparse.ArgumentParser(
        description="Generate TR-TAA adversarial images on PASCAL VOC"
    )
    parser.add_argument("--pascalvoc_root", type=str, required=True)
    parser.add_argument("--heads_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--K", type=int, required=True,
                        choices=[2, 4, 8, 16, 32],
                        help="Number of sampled transformations per PGD step")
    parser.add_argument("--attack_batch_size", type=int, default=4,
                        help="Batch size for attack generation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    # Attack hyperparameters (same as TAA, Pulfer et al. 2025)
    parser.add_argument("--eps_inf", type=float, default=8.0 / 255.0)
    parser.add_argument("--step_size", type=float, default=0.0004)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--feature_type", type=str, default="patch",
                        choices=["class", "patch", "class_and_patch"])
    parser.add_argument("--num_mean_samples", type=int, default=5000)
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration even if cached files exist")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed all RNGs and write to cached_images_clean_seed<seed>/ "
                             "(seed-stability runs). Default: unseeded, cached_images_clean/.")
    args = parser.parse_args()

    if args.seed is not None:
        from src.reproducibility import set_seed
        set_seed(args.seed)

    device = args.device
    model_info = get_model_info(args.model_name)

    # ============================================================
    # Set up directories
    # ============================================================
    cache_root = ("cached_images_clean" if args.seed is None
                  else f"cached_images_clean_seed{args.seed}")
    cache_dir = os.path.join("./outputs", cache_root, args.model_name)
    os.makedirs(cache_dir, exist_ok=True)

    cls_adv_path = os.path.join(cache_dir, f"cls_adv_images_K{args.K}.pt")
    seg_adv_path = os.path.join(cache_dir, f"seg_adv_images_K{args.K}.pt")
    cls_clean_path = os.path.join(cache_dir, "cls_clean_images.pt")
    cls_labels_path = os.path.join(cache_dir, "cls_labels.pt")
    seg_clean_path = os.path.join(cache_dir, "seg_clean_images.pt")
    seg_masks_path = os.path.join(cache_dir, "seg_masks.pt")
    psnr_path = os.path.join(cache_dir, f"psnr_info_K{args.K}.pt")
    mu_path = os.path.join(cache_dir, "mu_patch.pt")

    print(f"\n{'=' * 70}")
    print(f"GENERATING TR-TAA ADVERSARIAL IMAGES")
    print(f"Model: {args.model_name} ({model_info['arch']})")
    print(f"K (sampled transformations per PGD step): {args.K}")
    print(f"Feature type: {args.feature_type}")
    print(f"PGD: alpha={args.step_size}, steps={args.num_steps}")
    print(f"eps_inf = {args.eps_inf:.4f} ({args.eps_inf * 255:.0f}/255)")
    print(f"Attack batch size: {args.attack_batch_size}")
    print(f"Output directory: {cache_dir}")
    print(f"{'=' * 70}\n")

    # Check if output already exists
    if (os.path.exists(cls_adv_path) and os.path.exists(seg_adv_path)
            and not args.regenerate):
        print(f"Cached adversarial images for K={args.K} already exist:")
        print(f"  {cls_adv_path}")
        print(f"  {seg_adv_path}")
        print(f"Use --regenerate to overwrite. Exiting.")
        return

    # ============================================================
    # Load model and heads
    # ============================================================
    print("Loading model and trained heads...")
    backbone = get_dinov2_model(args.model_name, device)
    backbone.eval()

    # Heads are not needed for generation; loading them checks the checkpoints
    cls_head = LinearClassificationHead(model_info["dim"], 20).to(device)
    cls_head.load_state_dict(torch.load(
        os.path.join(args.heads_dir, f"{args.model_name}_classification_head.pth"),
        map_location=device,
        weights_only=False,
    ))
    cls_head.eval()

    seg_head = SegmentationHead(
        model_info["dim"], 21, model_info["patch_size"], args.image_size
    ).to(device)
    seg_head.load_state_dict(torch.load(
        os.path.join(args.heads_dir, f"{args.model_name}_segmentation_head.pth"),
        map_location=device,
        weights_only=False,
    ))
    seg_head.eval()

    # ============================================================
    # Compute or load mean features for centering
    # ============================================================
    if os.path.exists(mu_path):
        print(f"\nLoading cached mean vector from {mu_path}")
        mu = torch.load(mu_path, map_location=device, weights_only=False)
    else:
        print(f"\nComputing mean features ({args.num_mean_samples} samples)...")
        train_loader = get_pascalvoc_classification_loader(
            args.pascalvoc_root, "train", batch_size=64,
            num_workers=args.num_workers, image_size=args.image_size,
        )
        mu = compute_mean_features(backbone, train_loader, device,
                                    feature_type=args.feature_type,
                                    num_samples=args.num_mean_samples)
        torch.save(mu, mu_path)
        print(f"Mean vector saved to {mu_path}")

    # ============================================================
    # Load datasets
    # ============================================================
    print("\nLoading PascalVOC datasets...")
    cls_dataset = PascalVOCClassification(
        args.pascalvoc_root, split="val", image_size=args.image_size,
    )
    seg_dataset = PascalVOCSegmentation(
        args.pascalvoc_root, split="val", image_size=args.image_size,
    )

    cls_loader = torch.utils.data.DataLoader(
        cls_dataset, batch_size=args.attack_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    seg_loader = torch.utils.data.DataLoader(
        seg_dataset, batch_size=args.attack_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ============================================================
    # Generate TR-TAA on classification set
    # ============================================================
    print(f"\n--- Generating EoT-TAA (K={args.K}) on classification set ---")
    cls_adv_list = []
    cls_clean_list = []
    cls_labels_list = []
    cls_psnrs = []

    for images, labels, _ in tqdm(cls_loader, desc=f"EoT-TAA K={args.K} on cls"):
        images = images.to(device)
        adv_images = eot_taa_attack(
            images, backbone, mu,
            feature_type=args.feature_type,
            eps_inf=args.eps_inf,
            step_size=args.step_size,
            num_steps=args.num_steps,
            K=args.K,
            device=device,
            transform_reference=False,  # fixed clean reference
        )
        cls_adv_list.append(adv_images.cpu())
        cls_clean_list.append(images.cpu())
        cls_labels_list.append(labels)
        cls_psnrs.append(compute_psnr(images, adv_images))

    cls_adv_all = torch.cat(cls_adv_list, dim=0)
    cls_clean_all = torch.cat(cls_clean_list, dim=0)
    cls_labels_all = torch.cat(cls_labels_list, dim=0)
    avg_cls_psnr = float(np.mean(cls_psnrs))
    print(f"Generated {cls_adv_all.shape[0]} EoT-TAA images "
          f"(avg PSNR={avg_cls_psnr:.1f} dB)")

    # Save classification cache (clean and labels only need saving once)
    print(f"\nSaving classification cache to {cache_dir}...")
    torch.save(cls_adv_all, cls_adv_path)
    if not os.path.exists(cls_clean_path):
        torch.save(cls_clean_all, cls_clean_path)
    if not os.path.exists(cls_labels_path):
        torch.save(cls_labels_all, cls_labels_path)

    # Free memory before segmentation
    del cls_adv_list, cls_clean_list, cls_labels_list
    del cls_adv_all, cls_clean_all, cls_labels_all
    torch.cuda.empty_cache()

    # ============================================================
    # Generate TR-TAA on segmentation set
    # ============================================================
    print(f"\n--- Generating EoT-TAA (K={args.K}) on segmentation set ---")
    seg_adv_list = []
    seg_clean_list = []
    seg_masks_list = []
    seg_psnrs = []

    for images, masks, _ in tqdm(seg_loader, desc=f"EoT-TAA K={args.K} on seg"):
        images = images.to(device)
        adv_images = eot_taa_attack(
            images, backbone, mu,
            feature_type=args.feature_type,
            eps_inf=args.eps_inf,
            step_size=args.step_size,
            num_steps=args.num_steps,
            K=args.K,
            device=device,
            transform_reference=False,  # fixed clean reference
        )
        seg_adv_list.append(adv_images.cpu())
        seg_clean_list.append(images.cpu())
        seg_masks_list.append(masks)
        seg_psnrs.append(compute_psnr(images, adv_images))

    seg_adv_all = torch.cat(seg_adv_list, dim=0)
    seg_clean_all = torch.cat(seg_clean_list, dim=0)
    seg_masks_all = torch.cat(seg_masks_list, dim=0)
    avg_seg_psnr = float(np.mean(seg_psnrs))
    print(f"Generated {seg_adv_all.shape[0]} EoT-TAA images "
          f"(avg PSNR={avg_seg_psnr:.1f} dB)")

    # Save segmentation cache
    print(f"\nSaving segmentation cache to {cache_dir}...")
    torch.save(seg_adv_all, seg_adv_path)
    if not os.path.exists(seg_clean_path):
        torch.save(seg_clean_all, seg_clean_path)
    if not os.path.exists(seg_masks_path):
        torch.save(seg_masks_all, seg_masks_path)

    # Save PSNR info
    psnr_info = {"cls": avg_cls_psnr, "seg": avg_seg_psnr, "K": args.K}
    torch.save(psnr_info, psnr_path)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"K = {args.K}")
    print(f"Classification images: {cls_adv_path}")
    print(f"Segmentation  images: {seg_adv_path}")
    print(f"Avg PSNR (cls): {avg_cls_psnr:.1f} dB")
    print(f"Avg PSNR (seg): {avg_seg_psnr:.1f} dB")
    print(f"\nNext step: run evaluation script to measure drops under transforms.")


if __name__ == "__main__":
    main()
