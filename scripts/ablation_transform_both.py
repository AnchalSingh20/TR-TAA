"""
Ablation: clean-reference construction (fixed clean vs. transform-both).

Generates two TR-TAA variants under otherwise identical settings (backbone,
mu, eps, step size, steps, K, image subset); the only difference is the
reference feature:

    fixed clean (TR-TAA):  z_o = f(x_o) - mu        computed once
    transform-both:        z_o = f(t_k(x_o)) - mu   recomputed for every t_k

Both are evaluated under the nine evaluation transformations of
evaluate_eot_taa_clean.py on the first 500 classification and 500
segmentation images of PASCAL VOC val.

The classification numbers written by this script use the single-label
head; the multi-label mAP reported in the paper is obtained by re-scoring
the cached images with rescore_multilabel_all.py --target table13.
Segmentation (mIoU) numbers are used as written.

USAGE
    python scripts/ablation_transform_both.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14 --K 8 --n_cls 500 --n_seg 500
    python scripts/ablation_transform_both.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vitb14 --K 8 --n_cls 500 --n_seg 500

OUTPUTS
    outputs/cached_images_ablation/<model_name>/{cls,seg}_adv_{both,advonly}_K{K}.pt
    outputs/cached_images_ablation/<model_name>/{cls_clean,cls_labels,seg_clean,seg_masks}.pt
    outputs/results/ablation_transform_both_<model_name>_K{K}.txt / .pt
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
    get_dinov2_model,
    get_model_info,
    normalize_for_model,
    VFMFeatureExtractor,
)
from src.heads import LinearClassificationHead, SegmentationHead
from src.datasets import (
    PascalVOCClassification,
    PascalVOCSegmentation,
    get_pascalvoc_classification_loader,
)
from src.attacks import compute_mean_features, compute_psnr
# Reuse the EXACT transforms + evaluation used for the main results table,
# so ablation numbers are directly comparable.
from src.eot_attack import sample_random_transform
from evaluate_eot_taa_clean import evaluate_attack_set


# ============================================================
# TR-TAA attack parametrised by the reference construction
# ============================================================
def eot_taa_attack_variant(
    images,
    backbone,
    mu,
    transform_reference,        # True = transform-both ; False = transform-adv-only
    feature_type="patch",
    eps_inf=8.0 / 255.0,
    step_size=0.0004,
    num_steps=50,
    K=8,
    device="cuda",
):
    """
    Identical to src.eot_attack.eot_taa_attack EXCEPT for `transform_reference`.

    transform_reference=True  -> reference z_o is recomputed from t_k(x_o) for each k
    transform_reference=False -> reference z_o is computed once from the clean x_o
                                 and held fixed for all steps and all k (TR-TAA)
    """
    feature_extractor = VFMFeatureExtractor(backbone, feature_type)
    images = images.to(device)
    x_adv = images.clone().detach()

    # --- adv-only: precompute the FIXED clean reference once ---
    z_o_fixed = None
    if not transform_reference:
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
            t_k = sample_random_transform()

            x_adv_k = x_adv.clone()
            x_adv_k.requires_grad_(True)

            # transform the adversarial image (always)
            x_adv_transformed = t_k(x_adv_k)
            x_adv_t_norm = normalize_for_model(x_adv_transformed)
            z_adv = feature_extractor(x_adv_t_norm)

            if z_adv.dim() == 2:
                z_adv_tilde = z_adv - mu.unsqueeze(0)
            else:
                z_adv_tilde = z_adv - mu.unsqueeze(0).unsqueeze(0)
            z_adv_flat = z_adv_tilde.reshape(z_adv_tilde.shape[0], -1)

            # reference: transform-both vs fixed-clean
            if transform_reference:
                with torch.no_grad():
                    x_o_transformed = t_k(images)
                    x_o_t_norm = normalize_for_model(x_o_transformed)
                    z_o = feature_extractor(x_o_t_norm)
                    if z_o.dim() == 2:
                        z_o_tilde = z_o - mu.unsqueeze(0)
                    else:
                        z_o_tilde = z_o - mu.unsqueeze(0).unsqueeze(0)
                    z_o_flat = z_o_tilde.reshape(z_o_tilde.shape[0], -1)
            else:
                z_o_flat = z_o_fixed  # fixed clean reference

            cos_sim = F.cosine_similarity(z_adv_flat, z_o_flat, dim=-1).mean()

            grad_k = torch.autograd.grad(cos_sim, x_adv_k)[0]
            total_grad += grad_k.detach()

        avg_grad = total_grad / K
        x_adv = x_adv.detach() - step_size * avg_grad.sign()
        perturbation = torch.clamp(x_adv - images, min=-eps_inf, max=eps_inf)
        x_adv = torch.clamp(images + perturbation, min=0.0, max=1.0)

    # quantise (PNG storage simulation), matching eot_taa_attack
    x_adv = (x_adv.detach() * 255.0).round() / 255.0
    return torch.clamp(x_adv, 0.0, 1.0)


# ============================================================
# Generate one variant over a dataset subset
# ============================================================
def generate_variant(loader, backbone, mu, transform_reference, args, device, desc):
    adv_list, clean_list, tgt_list, psnrs = [], [], [], []
    for images, targets, _ in tqdm(loader, desc=desc):
        images = images.to(device)
        adv = eot_taa_attack_variant(
            images, backbone, mu,
            transform_reference=transform_reference,
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
    p = argparse.ArgumentParser(description="Reference-construction ablation: transform-both vs fixed clean")
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

    cache_dir = os.path.join("./outputs", "cached_images_ablation", args.model_name)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("./outputs/results", exist_ok=True)

    def cp(name):
        return os.path.join(cache_dir, name)

    print(f"\n{'=' * 72}")
    print("ABLATION: transform-both  vs  fixed clean reference (adv-only)")
    print(f"Model: {args.model_name} ({info['arch']})   K={args.K}")
    print(f"Subset: n_cls={args.n_cls}, n_seg={args.n_seg}   seed={args.seed}")
    print(f"{'=' * 72}\n")

    # ---- model + heads (heads needed only for evaluation) ----
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
    # GENERATE both variants (or load if cached)
    # ============================================================
    cls_both_p   = cp(f"cls_adv_both_K{args.K}.pt")
    cls_adv_p    = cp(f"cls_adv_advonly_K{args.K}.pt")
    seg_both_p   = cp(f"seg_adv_both_K{args.K}.pt")
    seg_adv_p    = cp(f"seg_adv_advonly_K{args.K}.pt")
    cls_clean_p  = cp("cls_clean.pt")
    cls_labels_p = cp("cls_labels.pt")
    seg_clean_p  = cp("seg_clean.pt")
    seg_masks_p  = cp("seg_masks.pt")

    have_all = all(os.path.exists(x) for x in
                   [cls_both_p, cls_adv_p, seg_both_p, seg_adv_p,
                    cls_clean_p, cls_labels_p, seg_clean_p, seg_masks_p])

    if have_all and not args.regenerate:
        print("\nCached ablation images found — loading (use --regenerate to overwrite).")
        cls_both   = torch.load(cls_both_p,   map_location="cpu", weights_only=False)
        cls_advo   = torch.load(cls_adv_p,    map_location="cpu", weights_only=False)
        seg_both   = torch.load(seg_both_p,   map_location="cpu", weights_only=False)
        seg_advo   = torch.load(seg_adv_p,    map_location="cpu", weights_only=False)
        cls_clean  = torch.load(cls_clean_p,  map_location="cpu", weights_only=False)
        cls_labels = torch.load(cls_labels_p, map_location="cpu", weights_only=False)
        seg_clean  = torch.load(seg_clean_p,  map_location="cpu", weights_only=False)
        seg_masks  = torch.load(seg_masks_p,  map_location="cpu", weights_only=False)
        psnr_both = psnr_advo = float("nan")
    else:
        print("\n[1/4] Generating transform-BOTH on classification subset...")
        cls_both, cls_clean, cls_labels, pcb = generate_variant(
            cls_loader, backbone, mu, True, args, device, "both/cls")
        print("[2/4] Generating transform-ADV-ONLY on classification subset...")
        cls_advo, _, _, pca = generate_variant(
            cls_loader, backbone, mu, False, args, device, "advonly/cls")
        print("[3/4] Generating transform-BOTH on segmentation subset...")
        seg_both, seg_clean, seg_masks, psb = generate_variant(
            seg_loader, backbone, mu, True, args, device, "both/seg")
        print("[4/4] Generating transform-ADV-ONLY on segmentation subset...")
        seg_advo, _, _, psa = generate_variant(
            seg_loader, backbone, mu, False, args, device, "advonly/seg")

        psnr_both = 0.5 * (pcb + psb)
        psnr_advo = 0.5 * (pca + psa)

        torch.save(cls_both, cls_both_p);   torch.save(cls_advo, cls_adv_p)
        torch.save(seg_both, seg_both_p);   torch.save(seg_advo, seg_adv_p)
        torch.save(cls_clean, cls_clean_p); torch.save(cls_labels, cls_labels_p)
        torch.save(seg_clean, seg_clean_p); torch.save(seg_masks, seg_masks_p)
        print(f"\nPSNR  transform-both: {psnr_both:.1f} dB | adv-only: {psnr_advo:.1f} dB")

    # ============================================================
    # EVALUATE both variants under the 10 shared transforms
    # ============================================================
    print(f"\n{'=' * 72}\nEvaluating: transform-BOTH\n{'=' * 72}")
    res_both = evaluate_attack_set(
        "transform-both",
        cls_both, cls_clean, cls_labels,
        seg_both, seg_clean, seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size)

    print(f"\n{'=' * 72}\nEvaluating: transform-ADV-ONLY\n{'=' * 72}")
    res_advo = evaluate_attack_set(
        "transform-adv-only",
        cls_advo, cls_clean, cls_labels,
        seg_advo, seg_clean, seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size)

    # ============================================================
    # Comparison table
    # ============================================================
    order = [r["transform"] for r in res_both]

    def get(res, tname):
        return next(r for r in res if r["transform"] == tname)

    rows = []
    for t in order:
        b, a = get(res_both, t), get(res_advo, t)
        rows.append([
            t,
            f"{b['cls_drop']:.1f}", f"{a['cls_drop']:.1f}",
            f"{b['seg_drop']:.1f}", f"{a['seg_drop']:.1f}",
        ])

    def avg(res, key):
        return np.mean([r[key] for r in res if r["transform"] != "No transform"])

    rows.append([
        "Avg (excl. clean)",
        f"{avg(res_both,'cls_drop'):.1f}", f"{avg(res_advo,'cls_drop'):.1f}",
        f"{avg(res_both,'seg_drop'):.1f}", f"{avg(res_advo,'seg_drop'):.1f}",
    ])

    headers = ["Transform",
               "both Cls↓", "advonly Cls↓",
               "both Seg↓", "advonly Seg↓"]

    table = tabulate(rows, headers=headers, tablefmt="grid")

    # averages over the nine transformations (excl. untransformed)
    b_cls, a_cls = avg(res_both, "cls_drop"), avg(res_advo, "cls_drop")
    b_seg, a_seg = avg(res_both, "seg_drop"), avg(res_advo, "seg_drop")
    d_cls, d_seg = b_cls - a_cls, b_seg - a_seg

    summary = (
        f"\n{'=' * 72}\n"
        f"REFERENCE-CONSTRUCTION ABLATION SUMMARY  ({args.model_name}, K={args.K}, "
        f"n_cls={args.n_cls}, n_seg={args.n_seg})\n"
        f"{'=' * 72}\n"
        f"Avg drop under 9 transforms (higher = more robust attack):\n"
        f"  Classification : both={b_cls:.1f}pp   adv-only={a_cls:.1f}pp   "
        f"Δ(both-advonly)={d_cls:+.1f}pp\n"
        f"  Segmentation   : both={b_seg:.1f}pp   adv-only={a_seg:.1f}pp   "
        f"Δ(both-advonly)={d_seg:+.1f}pp\n"
        f"PSNR: both={psnr_both:.1f} dB, adv-only={psnr_advo:.1f} dB\n"
    )

    print(f"\n{'=' * 72}")
    print("ABLATION RESULT: transform-both vs transform-adv-only "
          "(drop in pp, higher = stronger)")
    print(f"{'=' * 72}")
    print(table)
    print(summary)

    # ---- save ----
    txt = f"./outputs/results/ablation_transform_both_{args.model_name}_K{args.K}.txt"
    pt  = f"./outputs/results/ablation_transform_both_{args.model_name}_K{args.K}.pt"
    with open(txt, "w") as f:
        f.write(f"REFERENCE-CONSTRUCTION ABLATION: transform-both vs fixed clean (adv-only)\n")
        f.write(f"Model: {info['arch']}  K={args.K}  "
                f"n_cls={args.n_cls} n_seg={args.n_seg} seed={args.seed}\n")
        f.write("=" * 72 + "\n\n")
        f.write("Performance drop in pp (higher = stronger attack):\n")
        f.write(table)
        f.write("\n" + summary)
    torch.save({"both": res_both, "adv_only": res_advo,
                "avg": {"both_cls": b_cls, "advonly_cls": a_cls,
                        "both_seg": b_seg, "advonly_seg": a_seg}}, pt)

    print(f"Saved:\n  {txt}\n  {pt}")


if __name__ == "__main__":
    main()
