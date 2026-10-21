"""
Ablation: sensitivity to the operator application probabilities.

TR-TAA is generated with every base application probability of Table 1
halved (0.5x), unchanged (1.0x) and doubled (2.0x, each clamped to 1),
everything else held fixed (backbone, eps, step size, steps, K, image
subset, fixed clean reference). All conditions are evaluated under the
nine evaluation transformations on the first 500 classification and 500
segmentation images of PASCAL VOC val.

The classification numbers written by this script use the single-label
head; the multi-label mAP reported in the paper is obtained by re-scoring
the cached images with rescore_multilabel_all.py --target table15.
Segmentation (mIoU) numbers are used as written.

USAGE
    python scripts/ablation_transform_probability.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14 --K 8 --n_cls 500 --n_seg 500

OUTPUTS
    outputs/cached_images_ablation_prob/<model_name>/{cls,seg}_adv_<scale>_K{K}.pt
    outputs/cached_images_ablation_prob/<model_name>/{cls_clean,cls_labels,seg_clean,seg_masks}.pt
    outputs/results/ablation_transform_probability_<model_name>_K{K}.txt / .pt
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
from tqdm import tqdm
from tabulate import tabulate

# --- make project root AND scripts dir importable ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from src.models import get_dinov2_model, get_model_info
from src.heads import LinearClassificationHead, SegmentationHead
from src.datasets import (
    PascalVOCClassification,
    PascalVOCSegmentation,
    get_pascalvoc_classification_loader,
)
from src.attacks import compute_mean_features, compute_psnr
from src.eot_attack import eot_taa_attack, _BASE_PROB
# Reuse the EXACT evaluation used for the main results table, so ablation
# numbers are directly comparable.
from evaluate_eot_taa_clean import evaluate_attack_set


# ============================================================
# Conditions: name -> prob_scale
# ============================================================
CONDITIONS = [
    ("halved (0.5x)",   0.5),
    ("published (1.0x)", 1.0),
    ("doubled (2.0x)",  2.0),
]


def expected_num_transforms(prob_scale):
    """E[number of operators composed per sample], per-op clamped to 1.0."""
    return sum(min(1.0, base * prob_scale) for base in _BASE_PROB.values())


def slug(name):
    return name.split()[0]


# ============================================================
# Generate one condition over a dataset subset
# (fixed clean reference, as in TR-TAA)
# ============================================================
def generate_variant(loader, backbone, mu, prob_scale, args, device, desc):
    adv_list, clean_list, tgt_list, psnrs = [], [], [], []
    for images, targets, _ in tqdm(loader, desc=desc):
        images = images.to(device)
        adv = eot_taa_attack(
            images, backbone, mu,
            feature_type=args.feature_type,
            eps_inf=args.eps_inf, step_size=args.step_size,
            num_steps=args.num_steps, K=args.K, device=device,
            transform_reference=False,          # fixed clean reference (TR-TAA)
            transform_prob_scale=prob_scale,
        )
        adv_list.append(adv.cpu())
        clean_list.append(images.cpu())
        tgt_list.append(targets)
        psnrs.append(compute_psnr(images, adv))
    return (torch.cat(adv_list), torch.cat(clean_list),
            torch.cat(tgt_list), float(np.mean(psnrs)))


def main():
    p = argparse.ArgumentParser(description="Transform-probability sensitivity ablation for TR-TAA")
    p.add_argument("--pascalvoc_root", type=str, required=True)
    p.add_argument("--heads_dir", type=str, default="./checkpoints")
    p.add_argument("--model_name", type=str, default="dinov2_vits14")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n_cls", type=int, default=500,
                   help="number of classification images for the ablation subset")
    p.add_argument("--n_seg", type=int, default=500,
                   help="number of segmentation images for the ablation subset")
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

    cache_dir = os.path.join("./outputs", "cached_images_ablation_prob", args.model_name)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("./outputs/results", exist_ok=True)

    def cp(name):
        return os.path.join(cache_dir, name)

    print(f"\n{'=' * 72}")
    print("ABLATION: transform-PROBABILITY sensitivity (TR-TAA, fixed-clean reference)")
    print(f"Model: {args.model_name} ({info['arch']})   K={args.K}")
    print(f"Conditions: {[c[0] for c in CONDITIONS]}")
    for name, scale in CONDITIONS:
        print(f"  {name}: E[#transforms/sample] = {expected_num_transforms(scale):.2f}")
    print(f"Subset: n_cls={args.n_cls}, n_seg={args.n_seg}   seed={args.seed}")
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
                  f"(prob_scale={scale})...")
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

    avgs = {name: {"cls": avg(results[name], "cls_drop"),
                    "seg": avg(results[name], "seg_drop")}
            for name, _ in CONDITIONS}

    rows = [[name, f"{expected_num_transforms(scale):.2f}",
             f"{avgs[name]['cls']:.1f}", f"{avgs[name]['seg']:.1f}",
             f"{psnr[name]:.1f}"] for name, scale in CONDITIONS]
    headers = ["Condition", "E[#transforms]", "Avg Cls drop (pp)", "Avg Seg drop (pp)", "PSNR (dB)"]
    table = tabulate(rows, headers=headers, tablefmt="grid")

    pub_cls, pub_seg = avgs["published (1.0x)"]["cls"], avgs["published (1.0x)"]["seg"]
    half_cls, half_seg = avgs["halved (0.5x)"]["cls"], avgs["halved (0.5x)"]["seg"]
    dbl_cls, dbl_seg = avgs["doubled (2.0x)"]["cls"], avgs["doubled (2.0x)"]["seg"]
    max_swing_cls = max(abs(pub_cls - half_cls), abs(pub_cls - dbl_cls))
    max_swing_seg = max(abs(pub_seg - half_seg), abs(pub_seg - dbl_seg))

    summary_line = (f"max change vs. base probabilities: "
                    f"{max_swing_cls:.1f}pp (cls) / {max_swing_seg:.1f}pp (seg)")

    summary = (
        f"\n{'=' * 72}\n"
        f"TRANSFORM-PROBABILITY ABLATION SUMMARY  ({args.model_name}, K={args.K}, "
        f"n_cls={args.n_cls}, n_seg={args.n_seg})\n"
        f"{'=' * 72}\n"
        f"{summary_line}\n"
    )

    print(f"\n{'=' * 72}")
    print("ABLATION RESULT: transform-probability sensitivity (drop in pp, higher = stronger)")
    print(f"{'=' * 72}")
    print(table)
    print(summary)

    # ---- save ----
    txt = f"./outputs/results/ablation_transform_probability_{args.model_name}_K{args.K}.txt"
    pt = f"./outputs/results/ablation_transform_probability_{args.model_name}_K{args.K}.pt"
    with open(txt, "w") as f:
        f.write("TRANSFORM-PROBABILITY SENSITIVITY ABLATION (fixed clean reference)\n")
        f.write(f"Model: {info['arch']}  K={args.K}  "
                f"n_cls={args.n_cls} n_seg={args.n_seg} seed={args.seed}\n")
        f.write("=" * 72 + "\n\n")
        f.write(table)
        f.write("\n" + summary)
    torch.save({"results": results, "avg": avgs, "psnr": psnr}, pt)

    print(f"Saved:\n  {txt}\n  {pt}")


if __name__ == "__main__":
    main()
