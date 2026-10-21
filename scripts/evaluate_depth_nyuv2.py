"""
Evaluate monocular depth estimation on the NYU Depth v2 test set for clean,
TAA and TR-TAA (K=4, K=8) images, untransformed and under the nine
evaluation transformations or the input-preprocessing defenses.

The ground-truth depth maps are fixed; only the input image is attacked.
Predictions come from the frozen backbone + trained linear depth head
(log-depth, exponentiated to meters), evaluated with the standard NYUv2
metrics (Eigen et al. 2014) on the Eigen crop:
    RMSE (m), AbsRel   lower is better
    delta1             higher is better (fraction of pixels with
                       max(pred/gt, gt/pred) < 1.25)
The paper reports the attack-induced drop in delta1 (pp).

USAGE
    python scripts/evaluate_depth_nyuv2.py --nyuv2_root <NYUv2> \
        --model_name dinov2_vits14 --transforms
    python scripts/evaluate_depth_nyuv2.py --nyuv2_root <NYUv2> \
        --model_name dinov2_vitb14 --defenses     # JPEG/bit-depth/resize-pad/median
    python scripts/evaluate_depth_nyuv2.py --nyuv2_root <NYUv2> \
        --model_name dinov2_vitb14 --dither

OUTPUTS
    outputs/results/depth_nyuv2_<model_name>{,_defenses,_dither,...}.txt / .pt
"""

import os
import sys
import io
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from tqdm import tqdm
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_dinov2_model, get_model_info, normalize_for_model
from src.heads_depth import LinearDepthHead, log_depth_to_meters
from src.metrics_depth import compute_depth_metrics, average_metrics
from src.datasets_nyuv2 import NYUDepthV2BTS
from src.defenses import bit_depth_reduction, randomized_resize_pad, median_filter_defense, dither_defense, DiffPure


# ============================================================
# Evaluation transforms (same as classification/segmentation/retrieval)
# ============================================================

def apply_jpeg(images, quality=50):
    out = []
    for i in range(images.shape[0]):
        img = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(TF.to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(out).to(images.device)


def apply_blur(images, sigma=2.0):
    out = []
    for i in range(images.shape[0]):
        img = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        out.append(TF.to_tensor(img.filter(ImageFilter.GaussianBlur(radius=sigma))))
    return torch.stack(out).to(images.device)


def apply_resize(images, scale=0.5):
    B, C, H, W = images.shape
    small = F.interpolate(images, scale_factor=scale, mode="bilinear",
                          align_corners=False)
    return F.interpolate(small, size=(H, W), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def apply_rotation(images, angle=15):
    out = []
    for i in range(images.shape[0]):
        img = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        out.append(TF.to_tensor(TF.rotate(img, angle)))
    return torch.stack(out).to(images.device)


def apply_noise(images, sigma=0.03):
    return (images + torch.randn_like(images) * sigma).clamp(0, 1)


def apply_pipeline(images):
    images = apply_resize(images, scale=0.5)
    images = apply_blur(images, sigma=1.5)
    images = apply_noise(images, sigma=0.02)
    images = apply_jpeg(images, quality=60)
    return images


def get_eval_transforms():
    return {
        "No transform":   (lambda x: x, {}),
        "JPEG q=50":      (apply_jpeg,    {"quality": 50}),
        "JPEG q=30":      (apply_jpeg,    {"quality": 30}),
        "Blur s=1.0":     (apply_blur,    {"sigma": 1.0}),
        "Blur s=2.0":     (apply_blur,    {"sigma": 2.0}),
        "Resize 50%":     (apply_resize,  {"scale": 0.5}),
        "Resize 25%":     (apply_resize,  {"scale": 0.25}),
        "Rotation 15deg": (apply_rotation, {"angle": 15}),
        "Noise s=0.03":   (apply_noise,   {"sigma": 0.03}),
        "Print-scan":     (apply_pipeline, {}),
    }


def get_defense_transforms():
    """4-defense purification suite: JPEG + bit-depth + randomized
    resize/pad + median filter (spatial smoothing)."""
    return {
        "No defense":    (lambda x: x, {}),
        "JPEG q=50":     (apply_jpeg,               {"quality": 50}),
        "BitDepth b=4":  (bit_depth_reduction,       {"bits": 4}),
        "ResizePad":     (randomized_resize_pad,     {"min_scale": 0.9}),
        "Median k=3":    (median_filter_defense,     {"size": 3}),
    }


def get_jpeg_sweep_transforms():
    """JPEG-only quality sweep: realistic (75), aggressive-in-distribution
    (30), and out-of-distribution stress test (10)."""
    return {
        "No defense": (lambda x: x, {}),
        "JPEG q=75":  (apply_jpeg, {"quality": 75}),
        "JPEG q=30":  (apply_jpeg, {"quality": 30}),
        "JPEG q=10":  (apply_jpeg, {"quality": 10}),
    }


def get_dither_transforms():
    """Multi-level Floyd-Steinberg dithering only (recent purification
    defense, evaluated separately to avoid recomputing the other 4)."""
    return {
        "No defense":   (lambda x: x, {}),
        "DitherFS L=4": (dither_defense, {"levels": 4}),
    }


def get_diffpure_transforms(device, t_star=0.1):
    """Diffusion-based purification only (loads a UNet; constructed lazily
    so --defenses/--dither/--jpeg_sweep runs never pay this cost)."""
    dp = DiffPure(device=device, t_star=t_star)
    name = f"DiffPure t*={t_star} ({dp.loaded_model_id.split('/')[-1]})"
    return {
        "No defense": (lambda x: x, {}),
        name:         (lambda x: dp(x), {}),
    }


# ============================================================
# Depth prediction + per-image metrics
# ============================================================

@torch.no_grad()
def predict_depths(backbone, head, images, device,
                   min_depth=0.001, max_depth=10.0):
    """
    images: (B, 3, H, W) in [0,1]
    Returns: (B, 480, 640) depth in meters, clipped to valid range.
    """
    images = images.to(device)
    x = normalize_for_model(images)
    feats = backbone.forward_features(x)
    patch = feats["x_norm_patchtokens"]            # (B, N, D)
    log_pred = head(patch)                          # (B, 480, 640)
    return log_depth_to_meters(log_pred, min_depth=min_depth,
                                max_depth=max_depth)


def evaluate_under_transform(
    backbone, head, image_tensor, gt_depths, gt_valids,
    transform_func, t_kwargs, device, batch_size,
    min_depth=0.001, max_depth=10.0, desc="eval",
):
    """
    Apply transform to image_tensor, predict depth, compute per-image
    metrics, return averaged dict.

    Args:
        image_tensor: (N, 3, H, W) cached images (clean or adversarial)
        gt_depths: list of N (480, 640) tensors (in meters)
        gt_valids: list of N (480, 640) bool tensors
    """
    per_image = []
    N = image_tensor.shape[0]

    for s in tqdm(range(0, N, batch_size), desc=desc, leave=False):
        e = min(s + batch_size, N)
        batch = image_tensor[s:e].to(device)
        if transform_func is not None:
            batch = transform_func(batch, **(t_kwargs or {}))

        pred = predict_depths(backbone, head, batch, device,
                              min_depth=min_depth, max_depth=max_depth)
        pred_cpu = pred.cpu()

        for i in range(pred_cpu.shape[0]):
            global_idx = s + i
            m = compute_depth_metrics(
                pred_cpu[i], gt_depths[global_idx], gt_valids[global_idx],
                min_depth=min_depth, max_depth=max_depth,
            )
            per_image.append(m)

    return average_metrics(per_image)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate depth on NYUv2: clean vs TAA vs EoT-TAA"
    )
    parser.add_argument("--nyuv2_root", type=str, required=True)
    parser.add_argument("--heads_dir", type=str, default="./checkpoints")
    parser.add_argument("--cache_dir", type=str,
                        default="./outputs/cached_images_nyuv2")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min_depth", type=float, default=0.001)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--transforms", action="store_true",
                        help="Evaluate under all 10 transforms (slower)")
    parser.add_argument("--defenses", action="store_true",
                        help="Evaluate under the 4-defense purification suite "
                             "(JPEG/BitDepth/ResizePad/Median) instead of the "
                             "10-transform robustness suite.")
    parser.add_argument("--jpeg_sweep", action="store_true",
                        help="Evaluate under JPEG q={75,30,10} only.")
    parser.add_argument("--dither", action="store_true",
                        help="Evaluate under multi-level Floyd-Steinberg "
                             "dithering only.")
    parser.add_argument("--diffpure", action="store_true",
                        help="Evaluate under diffusion-based purification "
                             "(DiffPure) only.")
    parser.add_argument("--diffpure_tstar", type=float, default=0.1)
    args = parser.parse_args()

    device = args.device
    model_info = get_model_info(args.model_name)
    cache_dir = os.path.join(args.cache_dir, args.model_name)

    print(f"\n{'=' * 70}")
    print(f"NYU DEPTH v2 EVALUATION: clean vs TAA vs TR-TAA (EoT-TAA) K=4 vs K=8")
    print(f"Model: {args.model_name} ({model_info['arch']})")
    print(f"Metrics: RMSE(m), AbsRel, delta1 — attack stronger => RMSE up, d1 down")
    print(f"{'=' * 70}\n")

    # ---- Backbone ----
    backbone = get_dinov2_model(args.model_name, device)
    backbone.eval()

    # ---- Depth head ----
    head = LinearDepthHead(
        feature_dim=model_info["dim"],
        patch_size=model_info["patch_size"],
        image_size=args.image_size,
        output_size=(480, 640),
    ).to(device)

    head_path = os.path.join(
        args.heads_dir, f"{args.model_name}_depth_head_linear.pth"
    )
    if not os.path.exists(head_path):
        raise SystemExit(f"Depth head checkpoint not found: {head_path}")
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    print(f"Loaded depth head: {head_path}")
    if "val_metrics" in ckpt:
        v = ckpt["val_metrics"]
        print(f"  Training-time clean RMSE: {v['RMSE']:.4f}, "
              f"AbsRel: {v['AbsRel']:.4f}, d1: {v['delta1']:.4f}")

    # ---- Ground-truth depths (iterate the test dataset once) ----
    print("\nLoading ground-truth depths and valid masks...")
    test_set = NYUDepthV2BTS(
        args.nyuv2_root, split="test", image_size=args.image_size,
    )
    gt_depths, gt_valids = [], []
    for i in tqdm(range(len(test_set)), desc="GT depths", leave=False):
        _, d, v, _ = test_set[i]
        gt_depths.append(d)       # (480, 640) meters
        gt_valids.append(v)       # (480, 640) bool

    # ---- Image sets ----
    def load(p):
        return torch.load(p, map_location="cpu", weights_only=False)

    clean_path = os.path.join(cache_dir, "test_clean_images.pt")
    if not os.path.exists(clean_path):
        raise SystemExit(
            f"Clean cached images not found: {clean_path}\n"
            f"Run generate_eot_taa_nyuv2.py at least once first."
        )

    clean_imgs = load(clean_path)
    print(f"\n  clean images: {tuple(clean_imgs.shape)}")
    if clean_imgs.shape[0] != len(gt_depths):
        raise SystemExit(
            f"Mismatch: {clean_imgs.shape[0]} cached images vs "
            f"{len(gt_depths)} GT depths. Re-generate cleanly."
        )

    attack_sets = {"Clean": clean_imgs}

    taa_path = os.path.join(cache_dir, "test_adv_images_TAA.pt")
    if os.path.exists(taa_path):
        attack_sets["TAA"] = load(taa_path)
        print(f"  TAA images:   {tuple(attack_sets['TAA'].shape)}")

    for K in [4, 8]:
        kp = os.path.join(cache_dir, f"test_adv_images_K{K}.pt")
        if os.path.exists(kp):
            attack_sets[f"EoT-TAA K={K}"] = load(kp)
            print(f"  EoT-TAA K={K}: {tuple(attack_sets[f'EoT-TAA K={K}'].shape)}")

    # ---- Which transforms ----
    if args.diffpure:
        transforms_dict = get_diffpure_transforms(device, t_star=args.diffpure_tstar)
    elif args.dither:
        transforms_dict = get_dither_transforms()
    elif args.jpeg_sweep:
        transforms_dict = get_jpeg_sweep_transforms()
    elif args.defenses:
        transforms_dict = get_defense_transforms()
    elif args.transforms:
        transforms_dict = get_eval_transforms()
    else:
        transforms_dict = {"No transform": (lambda x: x, {})}

    # ---- Evaluate ----
    # all_results[attack_name][transform_name] = {RMSE, AbsRel, delta1, ...}
    all_results = {}

    for attack_name, imgs in attack_sets.items():
        all_results[attack_name] = {}
        print(f"\n{'=' * 70}\nEvaluating: {attack_name}\n{'=' * 70}")

        for t_name, (t_func, t_kwargs) in transforms_dict.items():
            metrics = evaluate_under_transform(
                backbone, head, imgs, gt_depths, gt_valids,
                t_func, t_kwargs, device, args.batch_size,
                min_depth=args.min_depth, max_depth=args.max_depth,
                desc=f"{attack_name}|{t_name}",
            )
            all_results[attack_name][t_name] = metrics
            print(f"  {t_name:18s}  "
                  f"RMSE={metrics['RMSE']:.4f}  "
                  f"AbsRel={metrics['AbsRel']:.4f}  "
                  f"d1={metrics['delta1']:.4f}")

    # ---- Tables ----
    baseline_t = next(iter(transforms_dict.keys()))  # "No transform" / "No defense"
    print(f"\n{'=' * 78}")
    print(f"NYU DEPTH v2 — '{baseline_t}' (input adversarial, GT depth fixed)")
    print(f"{'=' * 78}")
    rows = []
    for attack_name in all_results:
        r = all_results[attack_name][baseline_t]
        rows.append([
            attack_name,
            f"{r['RMSE']:.4f}",
            f"{r['AbsRel']:.4f}",
            f"{r['delta1']:.4f}",
        ])
    print(tabulate(rows, headers=["Attack", "RMSE↓", "AbsRel↓", "delta1↑"],
                   tablefmt="grid"))

    if "Clean" in all_results:
        base = all_results["Clean"][baseline_t]
        print(f"\n{'=' * 78}")
        print(f"ATTACK STRENGTH vs Clean ({baseline_t})")
        print(f"  Δ RMSE: positive = depth got worse (attack stronger)")
        print(f"  Δ d1:   negative = fewer accurate pixels (attack stronger)")
        print(f"{'=' * 78}")
        drow = []
        for attack_name in all_results:
            if attack_name == "Clean":
                continue
            r = all_results[attack_name][baseline_t]
            drow.append([
                attack_name,
                f"{r['RMSE'] - base['RMSE']:+.4f}",
                f"{r['AbsRel'] - base['AbsRel']:+.4f}",
                f"{r['delta1'] - base['delta1']:+.4f}",
            ])
        print(tabulate(drow,
                       headers=["Attack", "ΔRMSE", "ΔAbsRel", "Δdelta1"],
                       tablefmt="grid"))

    # Per-transform RMSE table (only if --transforms or --defenses)
    if args.transforms or args.defenses or args.jpeg_sweep or args.dither or args.diffpure:
        print(f"\n{'=' * 78}")
        print("RMSE (m, lower = better) under transforms")
        print(f"{'=' * 78}")
        tnames = list(transforms_dict.keys())
        headers = ["Transform"] + list(all_results.keys())
        trows = []
        for t in tnames:
            row = [t]
            for a in all_results:
                row.append(f"{all_results[a][t]['RMSE']:.4f}")
            trows.append(row)
        print(tabulate(trows, headers=headers, tablefmt="grid"))

        # Also delta1 table since RMSE alone doesn't tell the full story
        print(f"\n{'=' * 78}")
        print("delta1 (higher = better, fraction of correct-within-1.25x pixels)")
        print(f"{'=' * 78}")
        drows = []
        for t in tnames:
            row = [t]
            for a in all_results:
                row.append(f"{all_results[a][t]['delta1']:.4f}")
            drows.append(row)
        print(tabulate(drows, headers=headers, tablefmt="grid"))

    # ---- Save ----
    os.makedirs("./outputs/results", exist_ok=True)
    suffix = "_diffpure" if args.diffpure else ("_dither" if args.dither else ("_jpegsweep" if args.jpeg_sweep else ("_defenses" if args.defenses else "")))
    pt_path  = f"./outputs/results/depth_nyuv2_{args.model_name}{suffix}.pt"
    txt_path = f"./outputs/results/depth_nyuv2_{args.model_name}{suffix}.txt"
    torch.save(all_results, pt_path)

    with open(txt_path, "w") as f:
        f.write(f"NYU DEPTH v2 EVALUATION — Model: {model_info['arch']}\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"RMSE / AbsRel / delta1 ({baseline_t}):\n")
        f.write(tabulate(rows,
                         headers=["Attack", "RMSE", "AbsRel", "delta1"],
                         tablefmt="grid"))
        if "Clean" in all_results:
            f.write("\n\nAttack strength vs Clean (No transform):\n")
            f.write(tabulate(drow,
                             headers=["Attack", "ΔRMSE", "ΔAbsRel", "Δdelta1"],
                             tablefmt="grid"))
        if args.transforms or args.defenses or args.jpeg_sweep or args.dither or args.diffpure:
            f.write("\n\nRMSE under transforms:\n")
            f.write(tabulate(trows, headers=headers, tablefmt="grid"))
            f.write("\n\ndelta1 under transforms:\n")
            f.write(tabulate(drows, headers=headers, tablefmt="grid"))
    print(f"\nSaved:\n  {txt_path}\n  {pt_path}")


if __name__ == "__main__":
    main()
