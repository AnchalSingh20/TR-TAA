"""
Evaluate TAA and TR-TAA (K=4, K=8) on PASCAL VOC: multi-label
classification (mAP) and segmentation (mIoU), under the nine evaluation
transformations plus the untransformed setting.

This produces the classification and segmentation results of the main
tables and of the K ablation. It reads the cached adversarial images
written by generate_taa_baseline.py and generate_eot_taa_clean.py; no
attack is run.

Label alignment: the cached classification images were produced by
iterating PascalVOCClassification with shuffle=False. That dataset and
PascalVOCClassificationMultiLabel sort image ids identically and keep the
same images (every image with at least one positive class), so the i-th
cached image is the i-th multi-label sample. This is asserted with a
path-equality check on a sample of indices before evaluation.

USAGE
    python scripts/evaluate_eot_taa_multilabel.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14

OUTPUTS
    outputs/results/main_results_multilabel_<model_name>.txt / .pt
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

from src.models import get_dinov2_model, get_model_info
from src.heads import SegmentationHead
from src.heads_multilabel import MultiLabelClassificationHead
from src.metrics import (
    evaluate_segmentation_on_adversarial,
    confusion_to_miou,
)
from src.metrics_multilabel import (
    collect_scores_and_labels_on_adversarial,
    compute_map,
)
from src.datasets import PascalVOCClassification
from src.datasets_multilabel import PascalVOCClassificationMultiLabel


# ============================================================
# Evaluation transforms (same as evaluate_eot_taa_clean.py)
# ============================================================

def apply_jpeg(images, quality=50):
    result = []
    for i in range(images.shape[0]):
        img = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        result.append(TF.to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(result).to(images.device)


def apply_blur(images, sigma=2.0):
    result = []
    for i in range(images.shape[0]):
        img = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        result.append(TF.to_tensor(img.filter(ImageFilter.GaussianBlur(radius=sigma))))
    return torch.stack(result).to(images.device)


def apply_resize(images, scale=0.5):
    B, C, H, W = images.shape
    small = F.interpolate(images, scale_factor=scale, mode="bilinear",
                          align_corners=False)
    return F.interpolate(small, size=(H, W), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def apply_rotation(images, angle=15):
    result = []
    for i in range(images.shape[0]):
        img = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        result.append(TF.to_tensor(TF.rotate(img, angle)))
    return torch.stack(result).to(images.device)


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


# ============================================================
# Multi-label GT aligned by dataset-iteration order
# ============================================================

def build_aligned_multilabel_labels(pascalvoc_root, image_size, expected_N):
    """
    Build a (N, 20) multi-hot label tensor whose i-th row corresponds to
    the i-th image yielded by PascalVOCClassification (single-label, val).

    This relies on both single-label and multi-label datasets:
      - filtering by the same criterion (image has >=1 positive class)
      - sorting by image-id

    We assert this with a path check on a sample of indices.
    """
    sl_ds = PascalVOCClassification(
        root=pascalvoc_root, split="val", image_size=image_size, transform=None
    )
    ml_ds = PascalVOCClassificationMultiLabel(
        root=pascalvoc_root, split="val", image_size=image_size, transform=None
    )

    if len(ml_ds) != len(sl_ds):
        raise RuntimeError(
            f"Multi-label dataset has {len(ml_ds)} items but single-label "
            f"has {len(sl_ds)} — filtering criteria differ. Cannot align."
        )

    # Sanity: a sample of indices must agree on the image path.
    sample_idx = [0, len(sl_ds) // 4, len(sl_ds) // 2,
                  3 * len(sl_ds) // 4, len(sl_ds) - 1]
    for i in sample_idx:
        p_sl = sl_ds.images[i]
        p_ml = ml_ds.images[i]
        if p_sl != p_ml:
            raise RuntimeError(
                f"Dataset alignment broken at index {i}:\n"
                f"  single-label: {p_sl}\n"
                f"  multi-label:  {p_ml}\n"
                f"This means the two datasets produce different orderings."
            )

    print(f"  Alignment OK across {len(sample_idx)} sample indices.")

    labels = np.stack(ml_ds.labels, axis=0).astype(np.float32)   # (N, 20)
    labels_tensor = torch.from_numpy(labels)

    if labels_tensor.shape[0] >= expected_N:
        labels_tensor = labels_tensor[:expected_N]
    else:
        raise RuntimeError(
            f"Multi-label dataset is smaller than cache: "
            f"{labels_tensor.shape[0]} < {expected_N}"
        )

    return labels_tensor


# ============================================================
# Core per-attack evaluator with multi-label cls + mIoU seg
# ============================================================

def evaluate_attack_set_multilabel(
    attack_name,
    cls_adv, cls_clean, cls_labels_ml,
    seg_adv, seg_clean, seg_masks,
    backbone, cls_head, seg_head,
    device, eval_batch_size,
):
    transforms_dict = get_eval_transforms()
    results = []

    for t_name, (t_func, t_kwargs) in transforms_dict.items():
        # ---- Multi-label classification (mAP) ----
        adv_scores_list, adv_labels_list = [], []
        clean_scores_list, clean_labels_list = [], []

        for start in range(0, cls_adv.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, cls_adv.shape[0])
            batch_labels = cls_labels_ml[start:end]

            batch_adv = t_func(cls_adv[start:end].to(device), **t_kwargs)
            s, l = collect_scores_and_labels_on_adversarial(
                backbone, cls_head, batch_adv, batch_labels, device
            )
            adv_scores_list.append(s)
            adv_labels_list.append(l)
            del batch_adv

            batch_clean = t_func(cls_clean[start:end].to(device), **t_kwargs)
            s, l = collect_scores_and_labels_on_adversarial(
                backbone, cls_head, batch_clean, batch_labels, device
            )
            clean_scores_list.append(s)
            clean_labels_list.append(l)
            del batch_clean
            torch.cuda.empty_cache()

        adv_scores   = np.concatenate(adv_scores_list,   axis=0)
        adv_labels   = np.concatenate(adv_labels_list,   axis=0)
        clean_scores = np.concatenate(clean_scores_list, axis=0)
        clean_labels = np.concatenate(clean_labels_list, axis=0)

        cls_adv_map   = compute_map(adv_scores,   adv_labels)
        cls_clean_map = compute_map(clean_scores, clean_labels)
        cls_drop      = cls_clean_map - cls_adv_map

        # ---- Segmentation (mIoU, unchanged) ----
        adv_conf   = torch.zeros(21, 21, dtype=torch.long, device=device)
        clean_conf = torch.zeros(21, 21, dtype=torch.long, device=device)

        for start in range(0, seg_adv.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, seg_adv.shape[0])
            batch_masks = seg_masks[start:end]

            batch_adv = t_func(seg_adv[start:end].to(device), **t_kwargs)
            adv_conf += evaluate_segmentation_on_adversarial(
                backbone, seg_head, batch_adv, batch_masks, device
            )
            del batch_adv

            batch_clean = t_func(seg_clean[start:end].to(device), **t_kwargs)
            clean_conf += evaluate_segmentation_on_adversarial(
                backbone, seg_head, batch_clean, batch_masks, device
            )
            del batch_clean
            torch.cuda.empty_cache()

        seg_adv_miou   = confusion_to_miou(adv_conf)
        seg_clean_miou = confusion_to_miou(clean_conf)
        seg_drop       = seg_clean_miou - seg_adv_miou

        results.append({
            "transform":    t_name,
            "attack":       attack_name,
            "cls_clean":    cls_clean_map,
            "cls_adv":      cls_adv_map,
            "cls_drop":     cls_drop,
            "seg_clean":    seg_clean_miou,
            "seg_adv":      seg_adv_miou,
            "seg_drop":     seg_drop,
        })

        print(f"    {t_name:20s}  "
              f"Cls mAP: {cls_clean_map:.1f}→{cls_adv_map:.1f} (↓{cls_drop:.1f}pp)  "
              f"Seg mIoU: {seg_clean_miou:.1f}→{seg_adv_miou:.1f} (↓{seg_drop:.1f}pp)")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EoT-TAA with MULTI-LABEL classification (mAP)"
    )
    parser.add_argument("--pascalvoc_root", type=str, required=True)
    parser.add_argument("--heads_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--old_cache_dir", type=str, default="./outputs/cached_images",
        help="Directory containing the original TAA cache",
    )
    parser.add_argument(
        "--new_cache_dir", type=str, default="./outputs/cached_images_clean",
        help="Directory containing the EoT-TAA K=4 and K=8 caches",
    )
    args = parser.parse_args()

    device = args.device
    model_info = get_model_info(args.model_name)

    print(f"\n{'=' * 70}")
    print(f"MULTI-LABEL EVALUATION: TAA vs TR-TAA (EoT-TAA) K=4 vs K=8")
    print(f"Model: {args.model_name} ({model_info['arch']})")
    print(f"Cls metric: mAP   |   Seg metric: mIoU")
    print(f"{'=' * 70}\n")

    # ---------- Load model and heads ----------
    print("Loading model and heads...")
    backbone = get_dinov2_model(args.model_name, device)

    cls_head = MultiLabelClassificationHead(model_info["dim"], 20).to(device)
    cls_head_path = os.path.join(
        args.heads_dir,
        f"{args.model_name}_classification_head_multilabel.pth"
    )
    cls_head.load_state_dict(
        torch.load(cls_head_path, map_location=device, weights_only=False)
    )
    cls_head.eval()
    print(f"  Loaded multi-label cls head: {cls_head_path}")

    seg_head = SegmentationHead(
        model_info["dim"], 21, model_info["patch_size"], args.image_size
    ).to(device)
    seg_head_path = os.path.join(
        args.heads_dir, f"{args.model_name}_segmentation_head.pth"
    )
    seg_head.load_state_dict(
        torch.load(seg_head_path, map_location=device, weights_only=False)
    )
    seg_head.eval()
    print(f"  Loaded segmentation head: {seg_head_path}")

    # ---------- Load cached adversarial image sets ----------
    old_cache = os.path.join(args.old_cache_dir, args.model_name)
    new_cache = os.path.join(args.new_cache_dir, args.model_name)

    def load_pt(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    print("\nLoading cached image sets...")

    taa_cls_adv    = load_pt(os.path.join(old_cache, "cls_adv_images.pt"))
    taa_cls_clean  = load_pt(os.path.join(old_cache, "cls_clean_images.pt"))
    taa_seg_adv    = load_pt(os.path.join(old_cache, "seg_adv_images.pt"))
    taa_seg_clean  = load_pt(os.path.join(old_cache, "seg_clean_images.pt"))
    taa_seg_masks  = load_pt(os.path.join(old_cache, "seg_masks.pt"))
    print(f"  TAA:         cls={taa_cls_adv.shape[0]}, seg={taa_seg_adv.shape[0]}")

    k8_cls_adv    = load_pt(os.path.join(new_cache, "cls_adv_images_K8.pt"))
    k8_seg_adv    = load_pt(os.path.join(new_cache, "seg_adv_images_K8.pt"))
    eot_cls_clean = load_pt(os.path.join(new_cache, "cls_clean_images.pt"))
    eot_seg_clean = load_pt(os.path.join(new_cache, "seg_clean_images.pt"))
    eot_seg_masks = load_pt(os.path.join(new_cache, "seg_masks.pt"))
    print(f"  EoT-TAA K=8: cls={k8_cls_adv.shape[0]}, seg={k8_seg_adv.shape[0]}")

    k4_cls_adv = load_pt(os.path.join(new_cache, "cls_adv_images_K4.pt"))
    k4_seg_adv = load_pt(os.path.join(new_cache, "seg_adv_images_K4.pt"))
    print(f"  EoT-TAA K=4: cls={k4_cls_adv.shape[0]}, seg={k4_seg_adv.shape[0]}")

    # ---------- Build multi-label ground truth via dataset ordering ----------
    print("\nBuilding multi-label ground-truth (via aligned dataset order)...")
    cls_labels_ml = build_aligned_multilabel_labels(
        args.pascalvoc_root, args.image_size,
        expected_N=taa_cls_adv.shape[0],
    )
    print(f"  Multi-label GT shape: {cls_labels_ml.shape}  "
          f"(avg pos/img = {cls_labels_ml.sum(dim=1).mean().item():.2f})")

    # ---------- Sanity checks ----------
    assert taa_cls_adv.shape[0] == taa_cls_clean.shape[0] == cls_labels_ml.shape[0]
    assert k8_cls_adv.shape[0] == eot_cls_clean.shape[0]
    assert k4_cls_adv.shape[0] == eot_cls_clean.shape[0]

    # ---------- Evaluate each attack ----------
    all_results = {}

    print(f"\n{'=' * 70}")
    print("Evaluating: TAA (baseline)")
    print("=" * 70)
    all_results["TAA"] = evaluate_attack_set_multilabel(
        "TAA",
        taa_cls_adv, taa_cls_clean, cls_labels_ml[:taa_cls_adv.shape[0]],
        taa_seg_adv, taa_seg_clean, taa_seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size,
    )

    print(f"\n{'=' * 70}")
    print("Evaluating: EoT-TAA K=8")
    print("=" * 70)
    all_results["EoT-TAA K=8"] = evaluate_attack_set_multilabel(
        "EoT-TAA K=8",
        k8_cls_adv, eot_cls_clean, cls_labels_ml[:k8_cls_adv.shape[0]],
        k8_seg_adv, eot_seg_clean, eot_seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size,
    )

    print(f"\n{'=' * 70}")
    print("Evaluating: EoT-TAA K=4")
    print("=" * 70)
    all_results["EoT-TAA K=4"] = evaluate_attack_set_multilabel(
        "EoT-TAA K=4",
        k4_cls_adv, eot_cls_clean, cls_labels_ml[:k4_cls_adv.shape[0]],
        k4_seg_adv, eot_seg_clean, eot_seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size,
    )

    # ---------- Build the main results table ----------
    transforms_in_order = [r["transform"] for r in all_results["TAA"]]

    drop_rows = []
    for t_name in transforms_in_order:
        taa_r = next(r for r in all_results["TAA"]         if r["transform"] == t_name)
        k8_r  = next(r for r in all_results["EoT-TAA K=8"] if r["transform"] == t_name)
        k4_r  = next(r for r in all_results["EoT-TAA K=4"] if r["transform"] == t_name)
        drop_rows.append([
            t_name,
            f"{taa_r['cls_drop']:.1f}", f"{taa_r['seg_drop']:.1f}",
            f"{k4_r['cls_drop']:.1f}",  f"{k4_r['seg_drop']:.1f}",
            f"{k8_r['cls_drop']:.1f}",  f"{k8_r['seg_drop']:.1f}",
        ])

    def avg_drop(results, key):
        vals = [r[key] for r in results if r["transform"] != "No transform"]
        return float(np.mean(vals))

    drop_rows.append([
        "Avg (excl. clean)",
        f"{avg_drop(all_results['TAA'],         'cls_drop'):.1f}",
        f"{avg_drop(all_results['TAA'],         'seg_drop'):.1f}",
        f"{avg_drop(all_results['EoT-TAA K=4'], 'cls_drop'):.1f}",
        f"{avg_drop(all_results['EoT-TAA K=4'], 'seg_drop'):.1f}",
        f"{avg_drop(all_results['EoT-TAA K=8'], 'cls_drop'):.1f}",
        f"{avg_drop(all_results['EoT-TAA K=8'], 'seg_drop'):.1f}",
    ])

    headers = [
        "Transform",
        "TAA Cls↓", "TAA Seg↓",
        "K=4 Cls↓", "K=4 Seg↓",
        "K=8 Cls↓", "K=8 Seg↓",
    ]

    print(f"\n{'=' * 80}")
    print("MAIN RESULTS TABLE (MULTI-LABEL mAP / mIoU): Performance Drop in pp")
    print(f"{'=' * 80}")
    print(tabulate(drop_rows, headers=headers, tablefmt="grid"))

    # Improvement delta
    print(f"\n{'=' * 80}")
    print("IMPROVEMENT: EoT-TAA K=8 Δ over TAA")
    print(f"{'=' * 80}")
    delta_rows = []
    for t_name in transforms_in_order:
        taa_r = next(r for r in all_results["TAA"]         if r["transform"] == t_name)
        k8_r  = next(r for r in all_results["EoT-TAA K=8"] if r["transform"] == t_name)
        delta_rows.append([
            t_name,
            f"{k8_r['cls_drop'] - taa_r['cls_drop']:+.1f}",
            f"{k8_r['seg_drop'] - taa_r['seg_drop']:+.1f}",
        ])
    print(tabulate(delta_rows, headers=["Transform", "Δ Cls", "Δ Seg"], tablefmt="grid"))

    # Summary statistics
    taa_no_t = next(r for r in all_results["TAA"]         if r["transform"] == "No transform")
    k8_no_t  = next(r for r in all_results["EoT-TAA K=8"] if r["transform"] == "No transform")
    taa_avg_cls = avg_drop(all_results["TAA"],         "cls_drop")
    k8_avg_cls  = avg_drop(all_results["EoT-TAA K=8"], "cls_drop")
    taa_avg_seg = avg_drop(all_results["TAA"],         "seg_drop")
    k8_avg_seg  = avg_drop(all_results["EoT-TAA K=8"], "seg_drop")

    print(f"\n{'=' * 70}")
    print("SUMMARY (MULTI-LABEL)")
    print(f"{'=' * 70}")
    print(f"Clean baseline (no attack, no transform):")
    print(f"  Cls mAP:  {taa_no_t['cls_clean']:.2f}%")
    print(f"  Seg mIoU: {taa_no_t['seg_clean']:.2f}%")
    print(f"")
    print(f"TAA clean-domain drop:      Cls={taa_no_t['cls_drop']:.1f}pp, Seg={taa_no_t['seg_drop']:.1f}pp")
    print(f"EoT K=8 clean-domain drop:  Cls={k8_no_t['cls_drop']:.1f}pp, Seg={k8_no_t['seg_drop']:.1f}pp")
    print(f"TAA avg under transforms:   Cls={taa_avg_cls:.1f}pp, Seg={taa_avg_seg:.1f}pp")
    print(f"EoT K=8 avg under transforms: Cls={k8_avg_cls:.1f}pp, Seg={k8_avg_seg:.1f}pp")
    if taa_avg_cls > 0:
        print(f"Improvement factor (Cls avg): {k8_avg_cls / taa_avg_cls:.1f}×")
    if taa_avg_seg > 0:
        print(f"Improvement factor (Seg avg): {k8_avg_seg / taa_avg_seg:.1f}×")

    # ---------- Save ----------
    os.makedirs("./outputs/results", exist_ok=True)
    pt_path  = f"./outputs/results/main_results_multilabel_{args.model_name}.pt"
    txt_path = f"./outputs/results/main_results_multilabel_{args.model_name}.txt"

    torch.save(all_results, pt_path)

    with open(txt_path, "w") as f:
        f.write(f"MULTI-LABEL EVALUATION: TAA vs EoT-TAA (mAP / mIoU)\n")
        f.write(f"Model: {model_info['arch']}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Performance Drop in pp (higher = stronger attack):\n")
        f.write(tabulate(drop_rows, headers=headers, tablefmt="grid"))
        f.write("\n\nImprovement of EoT-TAA K=8 over TAA:\n")
        f.write(tabulate(delta_rows, headers=["Transform", "Δ Cls", "Δ Seg"], tablefmt="grid"))
        f.write(f"\n\nClean baseline: cls mAP={taa_no_t['cls_clean']:.2f}%, "
                f"seg mIoU={taa_no_t['seg_clean']:.2f}%\n")
        f.write(f"TAA clean-domain drop: cls={taa_no_t['cls_drop']:.1f}pp, "
                f"seg={taa_no_t['seg_drop']:.1f}pp\n")
        f.write(f"EoT K=8 clean-domain drop: cls={k8_no_t['cls_drop']:.1f}pp, "
                f"seg={k8_no_t['seg_drop']:.1f}pp\n")
        f.write(f"TAA avg under transforms: cls={taa_avg_cls:.1f}pp, "
                f"seg={taa_avg_seg:.1f}pp\n")
        f.write(f"EoT K=8 avg under transforms: cls={k8_avg_cls:.1f}pp, "
                f"seg={k8_avg_seg:.1f}pp\n")

    print(f"\nResults saved:")
    print(f"  {txt_path}")
    print(f"  {pt_path}")


if __name__ == "__main__":
    main()
