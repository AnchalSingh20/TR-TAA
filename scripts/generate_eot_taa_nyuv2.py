"""
Generate TAA or TR-TAA adversarial images on the NYU Depth v2 test set
(654 images, Eigen split).

    --taa_baseline   TAA (src.attacks.task_agnostic_attack), as for VOC
    --K <k>          TR-TAA with K sampled transformations per step and the
                     fixed clean reference

The centering mean mu is the PASCAL VOC mean of the same backbone
(outputs/cached_images_clean/<model_name>/mu_patch.pt, written by
generate_eot_taa_clean.py), so the same attack recipe is used on every task.

USAGE
    python scripts/generate_eot_taa_nyuv2.py --nyuv2_root <NYUv2> \
        --model_name dinov2_vits14 --taa_baseline
    python scripts/generate_eot_taa_nyuv2.py --nyuv2_root <NYUv2> \
        --model_name dinov2_vits14 --K 8
    python scripts/generate_eot_taa_nyuv2.py --nyuv2_root <NYUv2> \
        --model_name dinov2_vits14 --K 4

OUTPUTS (outputs/cached_images_nyuv2/<model_name>/)
    test_adv_images_{TAA,K4,K8}.pt, test_clean_images.pt,
    test_rel_paths.pt, psnr_nyuv2_{tag}.pt
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_dinov2_model, get_model_info
from src.attacks import compute_psnr, task_agnostic_attack
from src.eot_attack import eot_taa_attack
from src.datasets_nyuv2 import NYUDepthV2BTS


def main():
    parser = argparse.ArgumentParser(
        description="Generate TAA / TR-TAA on NYU Depth v2 test set"
    )
    parser.add_argument("--nyuv2_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--attack_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--K", type=int, default=None,
                        choices=[2, 4, 8, 16, 32],
                        help="Sampled transformations per PGD step (TR-TAA)")
    parser.add_argument("--taa_baseline", action="store_true",
                        help="TAA via task_agnostic_attack (no transform "
                             "sampling)")

    parser.add_argument("--eps_inf", type=float, default=8.0 / 255.0)
    parser.add_argument("--step_size", type=float, default=0.0004)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--feature_type", type=str, default="patch",
                        choices=["class", "patch", "class_and_patch"])
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    if (args.K is None) == (not args.taa_baseline):
        raise SystemExit(
            "Specify EXACTLY one of: --K <value>  OR  --taa_baseline"
        )

    # Fixed clean reference: t_k is applied to x_adv only (as for every task).
    os.environ["EOT_TRANSFORM_REFERENCE"] = "0"

    device = args.device
    model_info = get_model_info(args.model_name)

    if args.taa_baseline:
        tag = "TAA"
        method = "plain TAA (task_agnostic_attack)"
    else:
        tag = f"K{args.K}"
        method = f"TR-TAA (eot_taa_attack, K={args.K}, adv-only)"

    cache_dir = os.path.join("./outputs", "cached_images_nyuv2", args.model_name)
    os.makedirs(cache_dir, exist_ok=True)

    test_clean_path = os.path.join(cache_dir, "test_clean_images.pt")
    paths_path      = os.path.join(cache_dir, "test_rel_paths.pt")
    test_adv_path   = os.path.join(cache_dir, f"test_adv_images_{tag}.pt")
    psnr_path       = os.path.join(cache_dir, f"psnr_nyuv2_{tag}.pt")

    mu_cache_dir = os.path.join("./outputs", "cached_images_clean", args.model_name)
    mu_path = os.path.join(mu_cache_dir, "mu_patch.pt")

    print(f"\n{'=' * 70}")
    print(f"GENERATING {method}")
    print(f"ON NYU DEPTH v2 TEST SET")
    print(f"Model: {args.model_name} ({model_info['arch']})")
    print(f"Feature type: {args.feature_type}")
    print(f"PGD: alpha={args.step_size}, steps={args.num_steps}, "
          f"eps_inf={args.eps_inf:.4f} ({args.eps_inf*255:.0f}/255)")
    print(f"EOT_TRANSFORM_REFERENCE=0 (adv-only)")
    print(f"Output: {test_adv_path}")
    print(f"{'=' * 70}\n")

    if os.path.exists(test_adv_path) and not args.regenerate:
        print(f"Adversarial test set for tag={tag} already exists:")
        print(f"  {test_adv_path}")
        print("Use --regenerate to overwrite. Exiting.")
        return

    print("Loading backbone...")
    backbone = get_dinov2_model(args.model_name, device)
    backbone.eval()

    if not os.path.exists(mu_path):
        raise SystemExit(
            f"Cached mu not found at {mu_path}.\n"
            f"Run a PascalVOC generation first so mu is created. The SAME mu "
            f"must be used across all tasks for the cross-task argument."
        )
    print(f"Loading cached mean vector from {mu_path}")
    mu = torch.load(mu_path, map_location=device, weights_only=False)

    print("\nLoading NYU Depth v2 test set...")
    test_set = NYUDepthV2BTS(
        args.nyuv2_root, split="test", image_size=args.image_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.attack_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Test images: {len(test_set)}")

    print(f"\n--- Generating {tag} on NYUv2 test set ---")
    adv_list, clean_list, paths_list, psnrs = [], [], [], []

    for images, depths, valid_masks, rel_paths in tqdm(
        test_loader, desc=f"{tag} on NYUv2 test"
    ):
        images = images.to(device)

        if args.taa_baseline:
            # TAA: no transform sampling, as for classification/segmentation.
            adv_images = task_agnostic_attack(
                images, backbone, mu,
                feature_type=args.feature_type,
                eps_inf=args.eps_inf,
                step_size=args.step_size,
                num_steps=args.num_steps,
                device=device,
            )
        else:
            adv_images = eot_taa_attack(
                images, backbone, mu,
                feature_type=args.feature_type,
                eps_inf=args.eps_inf,
                step_size=args.step_size,
                num_steps=args.num_steps,
                K=args.K,
                device=device,
            )

        adv_list.append(adv_images.cpu())
        clean_list.append(images.cpu())
        paths_list.extend(list(rel_paths))
        psnrs.append(compute_psnr(images, adv_images))

    adv_all   = torch.cat(adv_list, dim=0)
    clean_all = torch.cat(clean_list, dim=0)
    avg_psnr  = float(np.mean(psnrs))
    print(f"Generated {adv_all.shape[0]} adversarial images "
          f"(avg PSNR={avg_psnr:.1f} dB)")

    print(f"\nSaving to {cache_dir} ...")
    torch.save(adv_all, test_adv_path)
    if not os.path.exists(test_clean_path):
        torch.save(clean_all, test_clean_path)
    if not os.path.exists(paths_path):
        torch.save(paths_list, paths_path)
    torch.save({"avg_psnr": avg_psnr, "per_batch": psnrs,
                "method": method}, psnr_path)

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"  Method:               {method}")
    print(f"  Adversarial test set: {test_adv_path}")
    print(f"  Clean test set:       {test_clean_path}")
    print(f"  Avg PSNR: {avg_psnr:.1f} dB")
    print(f"{'=' * 70}")
    print("Next: scripts/evaluate_depth_nyuv2.py --transforms")


if __name__ == "__main__":
    main()
