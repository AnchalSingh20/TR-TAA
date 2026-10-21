"""
Ablation: sensitivity to the operator parameter-range widths.

TR-TAA is generated with every operator's parameter range of Table 1
scaled around its centre by 0.5x (narrow), 1.0x (as in Table 1) and 1.5x
(wide), with the application probabilities unchanged, everything else
held fixed (backbone, eps, step size, steps, K, image subset, fixed clean
reference). All conditions are evaluated under the nine evaluation
transformations on the first 500 classification and 500 segmentation
images of PASCAL VOC val.

Implementation: scaling a range requires drawing the parameter before
deciding whether to apply the operator, which changes the order of RNG
calls relative to src.eot_attack.sample_random_transform. This script
therefore defines its own sampler and attack loop; the base
probabilities and the differentiable operators are imported from
src.eot_attack.

The classification numbers written by this script use the single-label
head; the multi-label mAP reported in the paper is obtained by re-scoring
the cached images with rescore_multilabel_all.py --target table15.
Segmentation (mIoU) numbers are used as written.

USAGE
    python scripts/ablation_transform_range.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14 --K 8 --n_cls 500 --n_seg 500

OUTPUTS
    outputs/cached_images_ablation_range/<model_name>/{cls,seg}_adv_<cond>_K{K}.pt
    outputs/cached_images_ablation_range/<model_name>/{cls_clean,cls_labels,seg_clean,seg_masks}.pt
    outputs/results/ablation_transform_range_<model_name>_K{K}.txt / .pt
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tabulate import tabulate

# --- make project root AND scripts dir importable ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from src.models import (
    get_dinov2_model, get_model_info, normalize_for_model, VFMFeatureExtractor,
)
from src.heads import LinearClassificationHead, SegmentationHead
from src.datasets import (
    PascalVOCClassification,
    PascalVOCSegmentation,
    get_pascalvoc_classification_loader,
)
from src.attacks import compute_mean_features, compute_psnr
from src.eot_attack import (
    diff_gaussian_blur, diff_resize, diff_jpeg_approx, diff_noise,
    diff_brightness, diff_contrast, diff_hflip, diff_rotation, _BASE_PROB,
)
# Reuse the EXACT evaluation used for the main results table, so ablation
# numbers are directly comparable.
from evaluate_eot_taa_clean import evaluate_attack_set


# ============================================================
# Parameter ranges of Table 1 and the per-operator bounds used to clamp
# scaled ranges.
# ============================================================
_BASE_RANGE = {
    "blur":       (0.5, 3.0),
    "resize":     (0.25, 0.75),
    "jpeg":       (30.0, 80.0),
    "noise":      (0.01, 0.05),
    "brightness": (0.7, 1.3),
    "contrast":   (0.7, 1.3),
    "rotation":   (-30.0, 30.0),
}
_RANGE_CLAMP = {   # (min_lo, max_hi); None = no clamp on that side
    "blur":       (0.05, None),
    "resize":     (0.05, 0.95),
    "jpeg":       (1.0, 100.0),
    "noise":      (0.0, None),
    "brightness": (0.05, None),
    "contrast":   (0.05, None),
    "rotation":   (None, None),
}


def scaled_range(name, range_scale):
    lo, hi = _BASE_RANGE[name]
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * range_scale
    lo2, hi2 = center - half, center + half
    cmin, cmax = _RANGE_CLAMP[name]
    if cmin is not None:
        lo2 = max(lo2, cmin)
    if cmax is not None:
        hi2 = min(hi2, cmax)
    return lo2, hi2


def sample_random_transform_ranged(range_scale):
    """
    Same 8-operator composition and PUBLISHED application probabilities as
    src.eot_attack.sample_random_transform, but samples each operator's
    parameter from a range whose WIDTH is scaled by `range_scale` around
    the published range's center.
    """
    transforms_to_apply = []

    if random.random() < _BASE_PROB["blur"]:
        lo, hi = scaled_range("blur", range_scale)
        sigma = random.uniform(lo, hi)
        transforms_to_apply.append(lambda x, s=sigma: diff_gaussian_blur(x, sigma=s))
    if random.random() < _BASE_PROB["resize"]:
        lo, hi = scaled_range("resize", range_scale)
        scale = random.uniform(lo, hi)
        transforms_to_apply.append(lambda x, s=scale: diff_resize(x, scale=s))
    if random.random() < _BASE_PROB["noise"]:
        lo, hi = scaled_range("noise", range_scale)
        sigma = random.uniform(lo, hi)
        transforms_to_apply.append(lambda x, s=sigma: diff_noise(x, sigma=s))
    if random.random() < _BASE_PROB["brightness"]:
        lo, hi = scaled_range("brightness", range_scale)
        factor = random.uniform(lo, hi)
        transforms_to_apply.append(lambda x, f=factor: diff_brightness(x, factor=f))
    if random.random() < _BASE_PROB["contrast"]:
        lo, hi = scaled_range("contrast", range_scale)
        factor = random.uniform(lo, hi)
        transforms_to_apply.append(lambda x, f=factor: diff_contrast(x, factor=f))
    if random.random() < _BASE_PROB["hflip"]:
        transforms_to_apply.append(lambda x: diff_hflip(x))
    if random.random() < _BASE_PROB["rotation"]:
        lo, hi = scaled_range("rotation", range_scale)
        angle = random.uniform(lo, hi)
        transforms_to_apply.append(lambda x, a=angle: diff_rotation(x, angle=a))
    if random.random() < _BASE_PROB["jpeg"]:
        lo, hi = scaled_range("jpeg", range_scale)
        quality = int(round(random.uniform(lo, hi)))
        transforms_to_apply.append(lambda x, q=quality: diff_jpeg_approx(x, quality=q))

    if len(transforms_to_apply) == 0:
        lo, hi = scaled_range("blur", range_scale)
        sigma = random.uniform(lo, hi)
        transforms_to_apply.append(lambda x, s=sigma: diff_gaussian_blur(x, sigma=s))

    def composed(x):
        for t in transforms_to_apply:
            x = t(x)
        return x

    return composed


# ============================================================
# TR-TAA attack with the fixed clean reference (as in
# ablation_transform_both.py), parametrised by range_scale.
# ============================================================
def eot_taa_attack_ranged(
    images, backbone, mu, range_scale,
    feature_type="patch", eps_inf=8.0 / 255.0, step_size=0.0004,
    num_steps=50, K=8, device="cuda",
):
    feature_extractor = VFMFeatureExtractor(backbone, feature_type)
    images = images.to(device)
    x_adv = images.clone().detach()

    with torch.no_grad():
        x_o_norm = normalize_for_model(images)
        z_o_fixed = feature_extractor(x_o_norm)
        if z_o_fixed.dim() == 2:
            z_o_fixed = z_o_fixed - mu.unsqueeze(0)
        else:
            z_o_fixed = z_o_fixed - mu.unsqueeze(0).unsqueeze(0)
        z_o_fixed = z_o_fixed.reshape(z_o_fixed.shape[0], -1)

    for _ in range(num_steps):
        total_grad = torch.zeros_like(images)
        for _k in range(K):
            t_k = sample_random_transform_ranged(range_scale)

            x_adv_k = x_adv.clone()
            x_adv_k.requires_grad_(True)

            x_adv_transformed = t_k(x_adv_k)
            x_adv_t_norm = normalize_for_model(x_adv_transformed)
            z_adv = feature_extractor(x_adv_t_norm)
            if z_adv.dim() == 2:
                z_adv_tilde = z_adv - mu.unsqueeze(0)
            else:
                z_adv_tilde = z_adv - mu.unsqueeze(0).unsqueeze(0)
            z_adv_flat = z_adv_tilde.reshape(z_adv_tilde.shape[0], -1)

            cos_sim = F.cosine_similarity(z_adv_flat, z_o_fixed, dim=-1).mean()
            grad_k = torch.autograd.grad(cos_sim, x_adv_k)[0]
            total_grad += grad_k.detach()

        avg_grad = total_grad / K
        x_adv = x_adv.detach() - step_size * avg_grad.sign()
        perturbation = torch.clamp(x_adv - images, min=-eps_inf, max=eps_inf)
        x_adv = torch.clamp(images + perturbation, min=0.0, max=1.0)

    x_adv = (x_adv.detach() * 255.0).round() / 255.0
    return torch.clamp(x_adv, 0.0, 1.0)


# ============================================================
# Conditions: name -> range_scale
# ============================================================
CONDITIONS = [
    ("narrow (0.5x width)",    0.5),
    ("published (1.0x width)", 1.0),
    ("wide (1.5x width)",      1.5),
]


def slug(name):
    return name.split()[0]


def print_ranges(range_scale):
    for name in _BASE_RANGE:
        lo, hi = scaled_range(name, range_scale)
        print(f"    {name:<12s}: [{lo:.3f}, {hi:.3f}]")


# ============================================================
# Generate one condition over a dataset subset
# ============================================================
def generate_variant(loader, backbone, mu, range_scale, args, device, desc):
    adv_list, clean_list, tgt_list, psnrs = [], [], [], []
    for images, targets, _ in tqdm(loader, desc=desc):
        images = images.to(device)
        adv = eot_taa_attack_ranged(
            images, backbone, mu, range_scale,
            feature_type=args.feature_type,
            eps_inf=args.eps_inf, step_size=args.step_size,
            num_steps=args.num_steps, K=args.K, device=device,
        )
        adv_list.append(adv.cpu())
        clean_list.append(images.cpu())
        tgt_list.append(targets)
        psnrs.append(compute_psnr(images, adv))
    return (torch.cat(adv_list), torch.cat(clean_list),
            torch.cat(tgt_list), float(np.mean(psnrs)))


def main():
    p = argparse.ArgumentParser(description="Transform range-width sensitivity ablation for TR-TAA")
    p.add_argument("--pascalvoc_root", type=str, required=True)
    p.add_argument("--heads_dir", type=str, default="./checkpoints")
    p.add_argument("--model_name", type=str, default="dinov2_vits14")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n_cls", type=int, default=500)
    p.add_argument("--n_seg", type=int, default=500)
    p.add_argument("--attack_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eps_inf", type=float, default=8.0 / 255.0)
    p.add_argument("--step_size", type=float, default=0.0004)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--feature_type", type=str, default="patch",
                   choices=["class", "patch", "class_and_patch"])
    p.add_argument("--num_mean_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regenerate", action="store_true")
    args = p.parse_args()

    device = args.device
    info = get_model_info(args.model_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cache_dir = os.path.join("./outputs", "cached_images_ablation_range", args.model_name)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("./outputs/results", exist_ok=True)

    def cp(name):
        return os.path.join(cache_dir, name)

    print(f"\n{'=' * 72}")
    print("ABLATION: parameter-RANGE-width sensitivity (TR-TAA, fixed-clean reference)")
    print(f"Model: {args.model_name} ({info['arch']})   K={args.K}")
    print(f"Subset: n_cls={args.n_cls}, n_seg={args.n_seg}   seed={args.seed}")
    for name, scale in CONDITIONS:
        print(f"  {name}:")
        print_ranges(scale)
    print(f"{'=' * 72}\n")

    # ---- model + heads ----
    print("Loading backbone and heads...")
    backbone = get_dinov2_model(args.model_name, device)
    backbone.eval()

    cls_head = LinearClassificationHead(info["dim"], 20).to(device)
    cls_head.load_state_dict(torch.load(
        os.path.join(args.heads_dir, f"{args.model_name}_classification_head_multilabel.pth"),
        map_location=device, weights_only=False))
    cls_head.eval()

    seg_head = SegmentationHead(info["dim"], 21, info["patch_size"], args.image_size).to(device)
    seg_head.load_state_dict(torch.load(
        os.path.join(args.heads_dir, f"{args.model_name}_segmentation_head.pth"),
        map_location=device, weights_only=False))
    seg_head.eval()

    # ---- mean vector (reuse the project's cached mu if present) ----
    mu_path = cp("mu_patch.pt")
    shared_mu = os.path.join("./outputs", "cached_images_clean", args.model_name, "mu_patch.pt")
    if os.path.exists(mu_path):
        mu = torch.load(mu_path, map_location=device, weights_only=False)
        print(f"Loaded ablation mu from {mu_path}")
    elif os.path.exists(shared_mu):
        mu = torch.load(shared_mu, map_location=device, weights_only=False)
        torch.save(mu, mu_path)
        print(f"Reused main-pipeline mu from {shared_mu}")
    else:
        print(f"Computing mean features ({args.num_mean_samples} samples)...")
        train_loader = get_pascalvoc_classification_loader(
            args.pascalvoc_root, "train", batch_size=64,
            num_workers=args.num_workers, image_size=args.image_size)
        mu = compute_mean_features(backbone, train_loader, device,
                                   feature_type=args.feature_type,
                                   num_samples=args.num_mean_samples)
        torch.save(mu, mu_path)

    # ---- datasets, subset to first n_cls / n_seg for speed ----
    print("Loading PascalVOC datasets (subset)...")
    cls_ds = PascalVOCClassification(args.pascalvoc_root, split="val", image_size=args.image_size)
    seg_ds = PascalVOCSegmentation(args.pascalvoc_root, split="val", image_size=args.image_size)
    cls_ds = torch.utils.data.Subset(cls_ds, range(min(args.n_cls, len(cls_ds))))
    seg_ds = torch.utils.data.Subset(seg_ds, range(min(args.n_seg, len(seg_ds))))

    cls_loader = torch.utils.data.DataLoader(
        cls_ds, batch_size=args.attack_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    seg_loader = torch.utils.data.DataLoader(
        seg_ds, batch_size=args.attack_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # ============================================================
    # GENERATE all three conditions (or load if cached)
    # ============================================================
    paths = {name: {
        "cls_adv": cp(f"cls_adv_{slug(name)}_K{args.K}.pt"),
        "seg_adv": cp(f"seg_adv_{slug(name)}_K{args.K}.pt"),
    } for name, _ in CONDITIONS}
    cls_clean_p, cls_labels_p = cp("cls_clean.pt"), cp("cls_labels.pt")
    seg_clean_p, seg_masks_p = cp("seg_clean.pt"), cp("seg_masks.pt")

    have_all = all(os.path.exists(paths[name][k]) for name, _ in CONDITIONS for k in ("cls_adv", "seg_adv")) \
        and all(os.path.exists(x) for x in [cls_clean_p, cls_labels_p, seg_clean_p, seg_masks_p])

    adv = {}
    psnr = {}
    if have_all and not args.regenerate:
        print("\nCached ablation images found — loading (use --regenerate to overwrite).")
        cls_clean = torch.load(cls_clean_p, map_location="cpu", weights_only=False)
        cls_labels = torch.load(cls_labels_p, map_location="cpu", weights_only=False)
        seg_clean = torch.load(seg_clean_p, map_location="cpu", weights_only=False)
        seg_masks = torch.load(seg_masks_p, map_location="cpu", weights_only=False)
        for name, _ in CONDITIONS:
            adv[name] = {
                "cls": torch.load(paths[name]["cls_adv"], map_location="cpu", weights_only=False),
                "seg": torch.load(paths[name]["seg_adv"], map_location="cpu", weights_only=False),
            }
            psnr[name] = float("nan")
    else:
        cls_clean = cls_labels = seg_clean = seg_masks = None
        for i, (name, scale) in enumerate(CONDITIONS, 1):
            print(f"\n[{i}/{len(CONDITIONS)}] Generating condition '{name}' "
                  f"(range_scale={scale})...")
            cls_adv, cc, cl, p_cls = generate_variant(
                cls_loader, backbone, mu, scale, args, device, f"{slug(name)}/cls")
            seg_adv, sc, sm, p_seg = generate_variant(
                seg_loader, backbone, mu, scale, args, device, f"{slug(name)}/seg")
            adv[name] = {"cls": cls_adv, "seg": seg_adv}
            psnr[name] = 0.5 * (p_cls + p_seg)
            torch.save(cls_adv, paths[name]["cls_adv"])
            torch.save(seg_adv, paths[name]["seg_adv"])
            if cls_clean is None:
                cls_clean, cls_labels, seg_clean, seg_masks = cc, cl, sc, sm
                torch.save(cls_clean, cls_clean_p); torch.save(cls_labels, cls_labels_p)
                torch.save(seg_clean, seg_clean_p); torch.save(seg_masks, seg_masks_p)
            print(f"  PSNR ({name}): {psnr[name]:.1f} dB")

    # ============================================================
    # EVALUATE all three conditions under the shared 9 transforms
    # ============================================================
    results = {}
    for name, _ in CONDITIONS:
        print(f"\n{'=' * 72}\nEvaluating: {name}\n{'=' * 72}")
        results[name] = evaluate_attack_set(
            name,
            adv[name]["cls"], cls_clean, cls_labels,
            adv[name]["seg"], seg_clean, seg_masks,
            backbone, cls_head, seg_head, device, args.eval_batch_size)

    # ============================================================
    # Comparison table
    # ============================================================
    def avg(res, key):
        return np.mean([r[key] for r in res if r["transform"] != "No transform"])

    def untransf(res, key):
        return next(r for r in res if r["transform"] == "No transform")[key]

    avgs = {name: {"cls": avg(results[name], "cls_drop"),
                    "seg": avg(results[name], "seg_drop")}
            for name, _ in CONDITIONS}
    untransf_vals = {name: {"cls": untransf(results[name], "cls_drop"),
                              "seg": untransf(results[name], "seg_drop")}
                      for name, _ in CONDITIONS}

    rows = [[name, f"{untransf_vals[name]['cls']:.1f}", f"{avgs[name]['cls']:.1f}",
             f"{untransf_vals[name]['seg']:.1f}", f"{avgs[name]['seg']:.1f}",
             f"{psnr[name]:.1f}"] for name, _ in CONDITIONS]
    headers = ["Condition", "Untransf. Cls", "Avg Cls drop", "Untransf. Seg", "Avg Seg drop", "PSNR (dB)"]
    table = tabulate(rows, headers=headers, tablefmt="grid")

    pub_cls, pub_seg = avgs["published (1.0x width)"]["cls"], avgs["published (1.0x width)"]["seg"]
    nar_cls, nar_seg = avgs["narrow (0.5x width)"]["cls"], avgs["narrow (0.5x width)"]["seg"]
    wide_cls, wide_seg = avgs["wide (1.5x width)"]["cls"], avgs["wide (1.5x width)"]["seg"]
    max_swing_cls = max(abs(pub_cls - nar_cls), abs(pub_cls - wide_cls))
    max_swing_seg = max(abs(pub_seg - nar_seg), abs(pub_seg - wide_seg))

    summary_line = (f"max change vs. Table 1 ranges: "
                    f"{max_swing_cls:.1f}pp (cls) / {max_swing_seg:.1f}pp (seg)")

    summary = (
        f"\n{'=' * 72}\n"
        f"TRANSFORM-RANGE ABLATION SUMMARY  ({args.model_name}, K={args.K}, "
        f"n_cls={args.n_cls}, n_seg={args.n_seg})\n"
        f"{'=' * 72}\n"
        f"{summary_line}\n"
    )

    print(f"\n{'=' * 72}")
    print("ABLATION RESULT: parameter-range-width sensitivity (drop in pp, higher = stronger)")
    print(f"{'=' * 72}")
    print(table)
    print(summary)

    # ---- save ----
    txt = f"./outputs/results/ablation_transform_range_{args.model_name}_K{args.K}.txt"
    pt = f"./outputs/results/ablation_transform_range_{args.model_name}_K{args.K}.pt"
    with open(txt, "w") as f:
        f.write("TRANSFORM-RANGE-WIDTH SENSITIVITY ABLATION (fixed clean reference)\n")
        f.write(f"Model: {info['arch']}  K={args.K}  "
                f"n_cls={args.n_cls} n_seg={args.n_seg} seed={args.seed}\n")
        f.write("=" * 72 + "\n\n")
        f.write(table)
        f.write("\n" + summary)
    torch.save({"results": results, "avg": avgs, "untransf": untransf_vals, "psnr": psnr}, pt)

    print(f"Saved:\n  {txt}\n  {pt}")


if __name__ == "__main__":
    main()
