"""
Cross-encoder (black-box) transfer between DINOv2 ViT-S/14 and ViT-B/14.

Loads the TAA and TR-TAA (K=4, K=8) images crafted on one backbone and
evaluates them (untransformed) with the other backbone and its own
independently trained heads. The attack never sees
the target encoder's weights or its centering mean. No new adversarial
images are generated.

Classification here uses the single-label head; the multi-label mAP
reported in the paper is computed by
rescore_multilabel_all.py --target cross_encoder. Segmentation (mIoU) is
used as written.

USAGE
    python scripts/evaluate_cross_encoder_transfer.py

OUTPUTS
    outputs/results/cross_encoder_transfer.txt / .pt
"""

import os
import sys
import argparse

import torch
from tabulate import tabulate

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from src.models import get_dinov2_model, get_model_info
from src.heads import LinearClassificationHead, SegmentationHead
from src.metrics import (
    evaluate_classification_on_adversarial,
    evaluate_segmentation_on_adversarial,
    confusion_to_miou,
)

MODELS = ["dinov2_vits14", "dinov2_vitb14"]
CONDITIONS = [("TAA", "cached_images", "cls_adv_images.pt", "seg_adv_images.pt"),
              ("EoT-TAA K=4", "cached_images_clean", "cls_adv_images_K4.pt", "seg_adv_images_K4.pt"),
              ("EoT-TAA K=8", "cached_images_clean", "cls_adv_images_K8.pt", "seg_adv_images_K8.pt")]


def load_model_and_heads(model_name, heads_dir, device):
    info = get_model_info(model_name)
    backbone = get_dinov2_model(model_name, device)
    backbone.eval()

    cls_head = LinearClassificationHead(info["dim"], 20).to(device)
    cls_head.load_state_dict(torch.load(
        os.path.join(heads_dir, f"{model_name}_classification_head_multilabel.pth"),
        map_location=device, weights_only=False))
    cls_head.eval()

    seg_head = SegmentationHead(info["dim"], 21, info["patch_size"], 518).to(device)
    seg_head.load_state_dict(torch.load(
        os.path.join(heads_dir, f"{model_name}_segmentation_head.pth"),
        map_location=device, weights_only=False))
    seg_head.eval()
    return backbone, cls_head, seg_head


def eval_cls(backbone, head, images, labels, device, batch_size=16):
    correct = total = 0
    for s in range(0, images.shape[0], batch_size):
        e = min(s + batch_size, images.shape[0])
        c, t = evaluate_classification_on_adversarial(
            backbone, head, images[s:e].to(device), labels[s:e], device)
        correct += c; total += t
        torch.cuda.empty_cache()
    return 100.0 * correct / total


def eval_seg(backbone, head, images, masks, device, batch_size=16):
    conf = torch.zeros(21, 21, dtype=torch.long, device=device)
    for s in range(0, images.shape[0], batch_size):
        e = min(s + batch_size, images.shape[0])
        conf += evaluate_segmentation_on_adversarial(
            backbone, head, images[s:e].to(device), masks[s:e], device)
        torch.cuda.empty_cache()
    return confusion_to_miou(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads_dir", type=str, default="./checkpoints")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    args = ap.parse_args()
    device = args.device

    print("Loading both backbones and heads...")
    models = {m: load_model_and_heads(m, args.heads_dir, device) for m in MODELS}

    print("Loading clean images/labels/masks (shared across backbones)...")
    cls_clean = torch.load("outputs/cached_images_clean/dinov2_vits14/cls_clean_images.pt",
                            map_location="cpu", weights_only=False)
    cls_labels = torch.load("outputs/cached_images_clean/dinov2_vits14/cls_labels.pt",
                             map_location="cpu", weights_only=False)
    seg_clean = torch.load("outputs/cached_images_clean/dinov2_vits14/seg_clean_images.pt",
                            map_location="cpu", weights_only=False)
    seg_masks = torch.load("outputs/cached_images_clean/dinov2_vits14/seg_masks.pt",
                            map_location="cpu", weights_only=False)

    # ---- White-box clean reference, computed once per target backbone ----
    clean_ref = {}
    for target in MODELS:
        backbone, cls_head, seg_head = models[target]
        print(f"\nComputing clean reference on {target}...")
        c = eval_cls(backbone, cls_head, cls_clean, cls_labels, device, args.eval_batch_size)
        s = eval_seg(backbone, seg_head, seg_clean, seg_masks, device, args.eval_batch_size)
        clean_ref[target] = {"cls": c, "seg": s}
        print(f"  Clean: cls={c:.2f}  seg={s:.2f}")

    results = []
    for source in MODELS:
        for cond_name, cache_dir, cls_file, seg_file in CONDITIONS:
            cls_path = os.path.join("outputs", cache_dir, source, cls_file)
            seg_path = os.path.join("outputs", cache_dir, source, seg_file)
            if not os.path.exists(cls_path):
                print(f"  [skip] missing {cls_path}")
                continue
            cls_adv = torch.load(cls_path, map_location="cpu", weights_only=False)
            seg_adv = torch.load(seg_path, map_location="cpu", weights_only=False)

            for target in MODELS:
                backbone, cls_head, seg_head = models[target]
                mode = "white-box" if source == target else "black-box transfer"
                print(f"\n{source} -> {target} ({mode}), {cond_name}")

                c_adv = eval_cls(backbone, cls_head, cls_adv, cls_labels, device, args.eval_batch_size)
                s_adv = eval_seg(backbone, seg_head, seg_adv, seg_masks, device, args.eval_batch_size)
                c_drop = clean_ref[target]["cls"] - c_adv
                s_drop = clean_ref[target]["seg"] - s_adv
                print(f"  cls: {clean_ref[target]['cls']:.2f} -> {c_adv:.2f} (drop {c_drop:.2f} pp)")
                print(f"  seg: {clean_ref[target]['seg']:.2f} -> {s_adv:.2f} (drop {s_drop:.2f} pp)")

                results.append({
                    "source": source, "target": target, "mode": mode,
                    "condition": cond_name,
                    "cls_clean": clean_ref[target]["cls"], "cls_adv": c_adv, "cls_drop": c_drop,
                    "seg_clean": clean_ref[target]["seg"], "seg_adv": s_adv, "seg_drop": s_drop,
                })

    # ---- table ----
    rows = [[r["source"], r["target"], r["mode"], r["condition"],
             f"{r['cls_drop']:.2f}", f"{r['seg_drop']:.2f}"] for r in results]
    table = tabulate(rows, headers=["Source", "Target", "Mode", "Condition",
                                     "Cls drop (pp)", "Seg drop (pp)"], tablefmt="grid")
    print("\n" + "=" * 72)
    print("CROSS-ENCODER TRANSFER RESULTS")
    print("=" * 72)
    print(table)

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/cross_encoder_transfer.txt", "w") as f:
        f.write("CROSS-ENCODER (BLACK-BOX) TRANSFER: ViT-S/14 <-> ViT-B/14\n")
        f.write("=" * 72 + "\n\n")
        f.write(table)
    torch.save(results, "outputs/results/cross_encoder_transfer.pt")
    print("\nSaved:\n  outputs/results/cross_encoder_transfer.txt\n  outputs/results/cross_encoder_transfer.pt")


if __name__ == "__main__":
    main()
