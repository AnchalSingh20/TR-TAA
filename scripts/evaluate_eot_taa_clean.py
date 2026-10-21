"""
Evaluate TAA and TR-TAA (K=4, K=8) on PASCAL VOC under the nine evaluation
transformations (plus the untransformed setting).

The evaluation transformations (get_eval_transforms) and the evaluation
loop (evaluate_attack_set) defined here are imported by the ablation and
other evaluation scripts.

The paper's main results (multi-label mAP and mIoU) are produced by
evaluate_eot_taa_multilabel.py. This script reports the same segmentation
(mIoU) numbers but scores classification with the single-label head and
top-1 accuracy, which is not used in the paper.

Inputs (cached images):
    TAA:         outputs/cached_images/<model>/        (generate_taa_baseline.py)
    TR-TAA K=4:  outputs/cached_images_clean/<model>/  (generate_eot_taa_clean.py --K 4)
    TR-TAA K=8:  outputs/cached_images_clean/<model>/  (generate_eot_taa_clean.py --K 8)

USAGE
    python scripts/evaluate_eot_taa_clean.py --pascalvoc_root <VOC2012> \
        --model_name dinov2_vits14

OUTPUTS
    outputs/results/main_results_clean.txt
    outputs/results/main_results_clean.pt
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
from src.heads import LinearClassificationHead, SegmentationHead
from src.metrics import (
    evaluate_classification_on_adversarial,
    evaluate_segmentation_on_adversarial,
    confusion_to_miou,
)


# ============================================================
# Evaluation transformations (nine conditions + untransformed)
# NON-differentiable: these are applied AFTER attack generation
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
# Core evaluation function
# Evaluate one set of adversarial images under all transforms
# ============================================================

def evaluate_attack_set(
    attack_name,
    cls_adv, cls_clean, cls_labels,
    seg_adv, seg_clean, seg_masks,
    backbone, cls_head, seg_head,
    device, eval_batch_size,
):
    """
    Evaluate one attack's adversarial images under all 10 transforms.
    Returns a list of dicts, one per transform.
    """
    transforms_dict = get_eval_transforms()
    results = []

    for t_name, (t_func, t_kwargs) in transforms_dict.items():
        # ---- Classification ----
        adv_correct = 0
        adv_total = 0
        clean_correct = 0
        clean_total = 0

        for start in range(0, cls_adv.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, cls_adv.shape[0])
            batch_labels = cls_labels[start:end]

            batch_adv = t_func(cls_adv[start:end].to(device), **t_kwargs)
            c, t = evaluate_classification_on_adversarial(
                backbone, cls_head, batch_adv, batch_labels, device)
            adv_correct += c
            adv_total += t
            del batch_adv

            batch_clean = t_func(cls_clean[start:end].to(device), **t_kwargs)
            c, t = evaluate_classification_on_adversarial(
                backbone, cls_head, batch_clean, batch_labels, device)
            clean_correct += c
            clean_total += t
            del batch_clean
            torch.cuda.empty_cache()

        cls_adv_acc   = 100.0 * adv_correct   / adv_total
        cls_clean_acc = 100.0 * clean_correct / clean_total
        cls_drop      = cls_clean_acc - cls_adv_acc

        # ---- Segmentation ----
        adv_conf   = torch.zeros(21, 21, dtype=torch.long, device=device)
        clean_conf = torch.zeros(21, 21, dtype=torch.long, device=device)

        for start in range(0, seg_adv.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, seg_adv.shape[0])
            batch_masks = seg_masks[start:end]

            batch_adv = t_func(seg_adv[start:end].to(device), **t_kwargs)
            adv_conf += evaluate_segmentation_on_adversarial(
                backbone, seg_head, batch_adv, batch_masks, device)
            del batch_adv

            batch_clean = t_func(seg_clean[start:end].to(device), **t_kwargs)
            clean_conf += evaluate_segmentation_on_adversarial(
                backbone, seg_head, batch_clean, batch_masks, device)
            del batch_clean
            torch.cuda.empty_cache()

        seg_adv_miou   = confusion_to_miou(adv_conf)
        seg_clean_miou = confusion_to_miou(clean_conf)
        seg_drop       = seg_clean_miou - seg_adv_miou

        results.append({
            "transform":    t_name,
            "attack":       attack_name,
            "cls_clean":    cls_clean_acc,
            "cls_adv":      cls_adv_acc,
            "cls_drop":     cls_drop,
            "seg_clean":    seg_clean_miou,
            "seg_adv":      seg_adv_miou,
            "seg_drop":     seg_drop,
        })

        print(f"    {t_name:20s}  "
              f"Cls: {cls_clean_acc:.1f}→{cls_adv_acc:.1f} (↓{cls_drop:.1f}pp)  "
              f"Seg: {seg_clean_miou:.1f}→{seg_adv_miou:.1f} (↓{seg_drop:.1f}pp)")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EoT-TAA clean pipeline — produce main results table"
    )
    parser.add_argument("--pascalvoc_root", type=str, required=True)
    parser.add_argument("--heads_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    model_info = get_model_info(args.model_name)

    print(f"\n{'=' * 70}")
    print(f"MAIN RESULTS EVALUATION: TAA vs TR-TAA (EoT-TAA) K=4 vs K=8")
    print(f"Model: {args.model_name} ({model_info['arch']})")
    print(f"{'=' * 70}\n")

    # ============================================================
    # Load model and heads
    # ============================================================
    print("Loading model and heads...")
    backbone = get_dinov2_model(args.model_name, device)

    cls_head = LinearClassificationHead(model_info["dim"], 20).to(device)
    cls_head.load_state_dict(torch.load(
        os.path.join(args.heads_dir, f"{args.model_name}_classification_head_multilabel.pth"),
        map_location=device, weights_only=False))
    cls_head.eval()

    seg_head = SegmentationHead(
        model_info["dim"], 21, model_info["patch_size"], args.image_size
    ).to(device)
    seg_head.load_state_dict(torch.load(
        os.path.join(args.heads_dir, f"{args.model_name}_segmentation_head.pth"),
        map_location=device, weights_only=False))
    seg_head.eval()

    # ============================================================
    # Load all three cached image sets
    # ============================================================
    old_cache = os.path.join("./outputs", "cached_images", args.model_name)
    new_cache = os.path.join("./outputs", "cached_images_clean", args.model_name)

    def load_pt(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    print("Loading cached image sets...")

    # TAA
    taa_cls_adv    = load_pt(os.path.join(old_cache, "cls_adv_images.pt"))
    taa_cls_clean  = load_pt(os.path.join(old_cache, "cls_clean_images.pt"))
    taa_cls_labels = load_pt(os.path.join(old_cache, "cls_labels.pt"))
    taa_seg_adv    = load_pt(os.path.join(old_cache, "seg_adv_images.pt"))
    taa_seg_clean  = load_pt(os.path.join(old_cache, "seg_clean_images.pt"))
    taa_seg_masks  = load_pt(os.path.join(old_cache, "seg_masks.pt"))
    print(f"  TAA:         cls={taa_cls_adv.shape[0]} imgs, "
          f"seg={taa_seg_adv.shape[0]} imgs  [old cache, verified]")

    # TR-TAA K=8
    k8_cls_adv    = load_pt(os.path.join(new_cache, "cls_adv_images_K8.pt"))
    k8_seg_adv    = load_pt(os.path.join(new_cache, "seg_adv_images_K8.pt"))
    # Clean images and labels are shared across K values (same dataset)
    eot_cls_clean  = load_pt(os.path.join(new_cache, "cls_clean_images.pt"))
    eot_cls_labels = load_pt(os.path.join(new_cache, "cls_labels.pt"))
    eot_seg_clean  = load_pt(os.path.join(new_cache, "seg_clean_images.pt"))
    eot_seg_masks  = load_pt(os.path.join(new_cache, "seg_masks.pt"))
    print(f"  EoT-TAA K=8: cls={k8_cls_adv.shape[0]} imgs, "
          f"seg={k8_seg_adv.shape[0]} imgs  [new clean cache]")

    # TR-TAA K=4
    k4_cls_adv = load_pt(os.path.join(new_cache, "cls_adv_images_K4.pt"))
    k4_seg_adv = load_pt(os.path.join(new_cache, "seg_adv_images_K4.pt"))
    print(f"  EoT-TAA K=4: cls={k4_cls_adv.shape[0]} imgs, "
          f"seg={k4_seg_adv.shape[0]} imgs  [new clean cache]")

    # ============================================================
    # Sanity check: dataset sizes must match within each attack
    # ============================================================
    assert taa_cls_adv.shape[0] == taa_cls_clean.shape[0] == taa_cls_labels.shape[0], \
        "TAA cls cache size mismatch"
    assert taa_seg_adv.shape[0] == taa_seg_clean.shape[0] == taa_seg_masks.shape[0], \
        "TAA seg cache size mismatch"
    assert k8_cls_adv.shape[0] == eot_cls_clean.shape[0], \
        "EoT K=8 cls size mismatch"
    assert k4_cls_adv.shape[0] == eot_cls_clean.shape[0], \
        "EoT K=4 cls size mismatch"

    # ============================================================
    # PSNR of the TR-TAA images
    # ============================================================
    psnr_k8_path = os.path.join(new_cache, "psnr_info_K8.pt")
    psnr_k4_path = os.path.join(new_cache, "psnr_info_K4.pt")
    if os.path.exists(psnr_k8_path):
        p = torch.load(psnr_k8_path, map_location="cpu", weights_only=False)
        print(f"\n  K=8 PSNR: cls={float(p['cls']):.1f} dB, seg={float(p['seg']):.1f} dB")
    if os.path.exists(psnr_k4_path):
        p = torch.load(psnr_k4_path, map_location="cpu", weights_only=False)
        print(f"  K=4 PSNR: cls={float(p['cls']):.1f} dB, seg={float(p['seg']):.1f} dB")

    # ============================================================
    # Run evaluation for each attack
    # ============================================================
    all_results = {}

    print(f"\n{'=' * 70}")
    print("Evaluating: TAA (baseline)")
    print("=" * 70)
    all_results["TAA"] = evaluate_attack_set(
        "TAA",
        taa_cls_adv, taa_cls_clean, taa_cls_labels,
        taa_seg_adv, taa_seg_clean, taa_seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size,
    )

    print(f"\n{'=' * 70}")
    print("Evaluating: EoT-TAA K=8")
    print("=" * 70)
    all_results["EoT-TAA K=8"] = evaluate_attack_set(
        "EoT-TAA K=8",
        k8_cls_adv, eot_cls_clean, eot_cls_labels,
        k8_seg_adv, eot_seg_clean, eot_seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size,
    )

    print(f"\n{'=' * 70}")
    print("Evaluating: EoT-TAA K=4")
    print("=" * 70)
    all_results["EoT-TAA K=4"] = evaluate_attack_set(
        "EoT-TAA K=4",
        k4_cls_adv, eot_cls_clean, eot_cls_labels,
        k4_seg_adv, eot_seg_clean, eot_seg_masks,
        backbone, cls_head, seg_head, device, args.eval_batch_size,
    )

    # ============================================================
    # Build the main results table
    # ============================================================
    transforms_in_order = [r["transform"] for r in all_results["TAA"]]

    # Paper-ready drop table
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

    # Add average row (excluding "No transform")
    def avg_drop(results, key):
        vals = [r[key] for r in results if r["transform"] != "No transform"]
        return np.mean(vals)

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

    # ============================================================
    # Print final tables
    # ============================================================
    print(f"\n{'=' * 80}")
    print("MAIN RESULTS TABLE: Performance Drop in pp (higher = stronger attack)")
    print(f"{'=' * 80}")
    print(tabulate(drop_rows, headers=headers, tablefmt="grid"))

    # Delta table (improvement of EoT-TAA over TAA)
    print(f"\n{'=' * 80}")
    print("IMPROVEMENT TABLE: EoT-TAA K=8 Δ over TAA")
    print(f"{'=' * 80}")
    delta_rows = []
    for t_name in transforms_in_order:
        taa_r = next(r for r in all_results["TAA"]         if r["transform"] == t_name)
        k8_r  = next(r for r in all_results["EoT-TAA K=8"] if r["transform"] == t_name)
        delta_cls = k8_r["cls_drop"] - taa_r["cls_drop"]
        delta_seg = k8_r["seg_drop"] - taa_r["seg_drop"]
        delta_rows.append([t_name, f"{delta_cls:+.1f}", f"{delta_seg:+.1f}"])
    print(tabulate(delta_rows, headers=["Transform", "Δ Cls", "Δ Seg"], tablefmt="grid"))

    # ============================================================
    # Summary statistics
    # ============================================================
    taa_no_t  = next(r for r in all_results["TAA"]         if r["transform"] == "No transform")
    k8_no_t   = next(r for r in all_results["EoT-TAA K=8"] if r["transform"] == "No transform")

    taa_avg_cls  = avg_drop(all_results["TAA"],         "cls_drop")
    k8_avg_cls   = avg_drop(all_results["EoT-TAA K=8"], "cls_drop")
    taa_avg_seg  = avg_drop(all_results["TAA"],         "seg_drop")
    k8_avg_seg   = avg_drop(all_results["EoT-TAA K=8"], "seg_drop")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"TAA clean-domain drop:    Cls={taa_no_t['cls_drop']:.1f}pp, "
          f"Seg={taa_no_t['seg_drop']:.1f}pp")
    print(f"EoT K=8 clean-domain drop: Cls={k8_no_t['cls_drop']:.1f}pp, "
          f"Seg={k8_no_t['seg_drop']:.1f}pp")
    print(f"TAA avg under transforms:  Cls={taa_avg_cls:.1f}pp, Seg={taa_avg_seg:.1f}pp")
    print(f"EoT K=8 avg under transforms: Cls={k8_avg_cls:.1f}pp, Seg={k8_avg_seg:.1f}pp")
    if taa_avg_cls > 0:
        improvement = k8_avg_cls / taa_avg_cls
        print(f"Improvement factor (Cls avg): {improvement:.1f}×")
    if taa_avg_seg > 0:
        improvement_seg = k8_avg_seg / taa_avg_seg
        print(f"Improvement factor (Seg avg): {improvement_seg:.1f}×")

    # ============================================================
    # Save results
    # ============================================================
    os.makedirs("./outputs/results", exist_ok=True)
    pt_path  = "./outputs/results/main_results_clean.pt"
    txt_path = "./outputs/results/main_results_clean.txt"

    torch.save(all_results, pt_path)

    with open(txt_path, "w") as f:
        f.write("MAIN RESULTS: TAA vs EoT-TAA (Clean Pipeline)\n")
        f.write(f"Model: {model_info['arch']}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Performance Drop in pp (higher = stronger attack):\n")
        f.write(tabulate(drop_rows, headers=headers, tablefmt="grid"))
        f.write("\n\nImprovement of EoT-TAA K=8 over TAA:\n")
        f.write(tabulate(delta_rows,
                          headers=["Transform", "Δ Cls", "Δ Seg"],
                          tablefmt="grid"))
        f.write(f"\n\nTAA clean-domain: cls={taa_no_t['cls_drop']:.1f}pp, "
                f"seg={taa_no_t['seg_drop']:.1f}pp\n")
        f.write(f"EoT K=8 clean-domain: cls={k8_no_t['cls_drop']:.1f}pp, "
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
