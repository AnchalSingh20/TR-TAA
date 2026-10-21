"""
Evaluate clean / TAA / TR-TAA (K=4, K=8) PASCAL VOC images under
input-preprocessing defenses, DINOv2 ViT-B/14.

Defenses (src/defenses.py): JPEG recompression (one column per
--jpeg_qualities value), bit-depth reduction (4 bits), randomized
resize-and-pad (scale in [0.9, 1.0]), median filter (3x3) and
Floyd-Steinberg dithering (4 levels). Each can be switched off with
--no_<defense>. DiffPure is optional and not used in the paper
(--no_diffpure).

The segmentation (mIoU) numbers of this script are used in the paper. Its
classification column is single-label accuracy; the multi-label mAP under
the same defenses is computed by rescore_multilabel_all.py --target table16
(and --target jpegsweep_cls for the JPEG sweep).

Inputs: clean and TR-TAA images from --cache_dir
(outputs/cached_images_clean/<model>), TAA images from --taa_cache_dir
(outputs/cached_images/<model>). Both caches iterate the same VOC val
loader with shuffle=False, so they are index-aligned (checked at startup).

USAGE (the three runs used in the paper)
    C="--model_name dinov2_vitb14 --cache_dir ./outputs/cached_images_clean/dinov2_vitb14 \
       --taa_cache_dir ./outputs/cached_images/dinov2_vitb14 --no_diffpure"
    python scripts/evaluate_defenses_voc.py $C --jpeg_qualities 90 75 50 30 10 \
        --no_bitdepth --no_resize_pad --no_median --no_dither --tag jpegsweep
    python scripts/evaluate_defenses_voc.py $C --jpeg_qualities 75 --no_dither \
        --tag defenses4_full
    python scripts/evaluate_defenses_voc.py $C --jpeg_qualities 75 \
        --no_bitdepth --no_resize_pad --no_median --tag dither_full

OUTPUTS
    outputs/results/defenses_voc_<model_name>_<tag>.txt / .pt
"""

import os
import sys
import argparse
import torch
import numpy as np
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
from src.defenses import build_defenses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="dinov2_vitb14")
    ap.add_argument("--cache_dir", type=str,
                    default="./outputs/cached_images_clean/dinov2_vitb14",
                    help="Where Clean + EoT-TAA K={4,8} adversarial .pt sets live "
                         "(from generate_eot_taa_clean.py).")
    ap.add_argument("--taa_cache_dir", type=str,
                    default="./outputs/cached_images/dinov2_vitb14",
                    help="Where the TAA-baseline adversarial .pt sets live "
                         "(from evaluate_taa_survival.py). The TAA row is added "
                         "if files exist; silently skipped otherwise.")
    ap.add_argument("--heads_dir", type=str, default="./checkpoints")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_images", type=int, default=None,
                    help="Cap images per condition (quick sanity pass).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="",
                    help="Free-form suffix appended to output filenames so "
                         "successive runs (smoke / jpeg-only / diffpure) don't "
                         "overwrite each other. E.g. --tag smoke")

    # ---- defense knobs ----
    # Backward-compatible: --jpeg_quality (single int) still works. Prefer
    # --jpeg_qualities (list) for several JPEG columns in one run.
    ap.add_argument("--jpeg_quality", type=int, default=75,
                    help="Single JPEG quality. Use --jpeg_qualities "
                         "to set multiple.")
    ap.add_argument("--jpeg_qualities", type=int, nargs="+", default=None,
                    help="One or more JPEG qualities (e.g. --jpeg_qualities 75 50). "
                         "Overrides --jpeg_quality if given.")
    ap.add_argument("--no_bitdepth", action="store_true")
    ap.add_argument("--bit_depth", type=int, default=4)
    ap.add_argument("--no_resize_pad", action="store_true")
    ap.add_argument("--resize_pad_min_scale", type=float, default=0.9)
    ap.add_argument("--no_median", action="store_true")
    ap.add_argument("--median_size", type=int, default=3)
    ap.add_argument("--no_dither", action="store_true")
    ap.add_argument("--dither_levels", type=int, default=4)
    ap.add_argument("--no_diffpure", action="store_true")
    ap.add_argument("--diffpure_tstar", type=float, default=0.1)
    ap.add_argument("--diffpure_steps", type=int, default=None,
                    help="Subsample reverse DDPM steps for speed (e.g. 25). "
                         "Default None = full sequential from t* to 0.")
    ap.add_argument("--diffpure_model", type=str, default="",
                    help="Force a specific diffusers model id (else auto).")
    args = ap.parse_args()

    # Normalize JPEG knob into a single canonical list on `args`.
    if args.jpeg_qualities is None:
        args.jpeg_qualities = [args.jpeg_quality]

    device = args.device
    info = get_model_info(args.model_name)

    print(f"\n{'=' * 72}")
    print("PASCALVOC DEFENSES EVALUATION (classification + segmentation)")
    print(f"Backbone: {args.model_name} ({info['arch']})")
    print(f"Defenses: No-defense, "
          f"JPEG q={args.jpeg_qualities}"
          f"{'' if args.no_diffpure else f', DiffPure t*={args.diffpure_tstar}'}")
    print(f"{'=' * 72}\n")

    # ---- Backbone ----
    print("Loading backbone...")
    backbone = get_dinov2_model(args.model_name, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ---- Heads (NOTE the _multilabel classification head filename) ----
    cls_head = LinearClassificationHead(info["dim"], 20).to(device)
    cls_path = os.path.join(args.heads_dir,
                            f"{args.model_name}_classification_head_multilabel.pth")
    cls_head.load_state_dict(torch.load(cls_path, map_location=device))
    cls_head.eval()
    print(f"  cls head: {cls_path}")

    seg_head = SegmentationHead(
        info["dim"], 21, info["patch_size"], args.image_size
    ).to(device)
    seg_path = os.path.join(args.heads_dir,
                            f"{args.model_name}_segmentation_head.pth")
    seg_head.load_state_dict(torch.load(seg_path, map_location=device))
    seg_head.eval()
    print(f"  seg head: {seg_path}")

    # ---- Cached images ----
    def load_main(name):
        p = os.path.join(args.cache_dir, name)
        if not os.path.exists(p):
            return None
        return torch.load(p, map_location="cpu", weights_only=False)

    def load_taa(name):
        p = os.path.join(args.taa_cache_dir, name)
        if not os.path.exists(p):
            return None
        return torch.load(p, map_location="cpu", weights_only=False)

    cls_labels = load_main("cls_labels.pt")
    seg_masks  = load_main("seg_masks.pt")
    if cls_labels is None or seg_masks is None:
        raise SystemExit("Missing cls_labels.pt or seg_masks.pt in --cache_dir.")

    # ---- TAA images live in --taa_cache_dir (generate_taa_baseline.py writes
    # cls_adv_images.pt / seg_adv_images.pt to outputs/cached_images/<model>/).
    # If they are absent, the TAA row is skipped.
    cls_sets = {
        "Clean":       load_main("cls_clean_images.pt"),
        "TAA":         load_taa("cls_adv_images.pt"),
        "EoT-TAA K=4": load_main("cls_adv_images_K4.pt"),
        "EoT-TAA K=8": load_main("cls_adv_images_K8.pt"),
    }
    seg_sets = {
        "Clean":       load_main("seg_clean_images.pt"),
        "TAA":         load_taa("seg_adv_images.pt"),
        "EoT-TAA K=4": load_main("seg_adv_images_K4.pt"),
        "EoT-TAA K=8": load_main("seg_adv_images_K8.pt"),
    }

    # Only keep conditions where BOTH cls and seg tensors are present.
    conditions = [k for k in cls_sets
                  if cls_sets[k] is not None and seg_sets[k] is not None]
    missing = [k for k in cls_sets if k not in conditions]
    print(f"  conditions found: {conditions}")
    if missing:
        print(f"  conditions skipped (no cache): {missing}")
    if "TAA" not in conditions:
        print(f"  NOTE: TAA row absent. Looked in: {args.taa_cache_dir}\n"
              f"        Expected files: cls_adv_images.pt, seg_adv_images.pt\n"
              f"        (Generated by generate_taa_baseline.py.)")

    # ---- Optional truncation for a quick pass ----
    def maybe_trunc(t):
        if t is None or args.max_images is None:
            return t
        return t[:args.max_images]

    cls_labels = maybe_trunc(cls_labels)
    seg_masks  = maybe_trunc(seg_masks)
    for d in (cls_sets, seg_sets):
        for k in list(d.keys()):
            d[k] = maybe_trunc(d[k])

    n_cls = cls_labels.shape[0]
    n_seg = seg_masks.shape[0]
    print(f"  evaluating {n_cls} cls images, {n_seg} seg images "
          f"({'truncated' if args.max_images else 'full'})\n")

    # ---- Alignment check: every condition must have matching tensor lengths.
    # If TAA was generated against a different split / image_size / sort order,
    # we want a hard error here rather than a silently wrong table.
    for cond in conditions:
        if cls_sets[cond].shape[0] != n_cls:
            raise SystemExit(
                f"Size mismatch: cls_sets['{cond}'] has "
                f"{cls_sets[cond].shape[0]} images, expected {n_cls}.\n"
                f"For TAA: regenerate so it matches the SAME PascalVOC val "
                f"split / image_size={args.image_size} / sort order as the "
                f"EoT-TAA caches in {args.cache_dir}.")
        if seg_sets[cond].shape[0] != n_seg:
            raise SystemExit(
                f"Size mismatch: seg_sets['{cond}'] has "
                f"{seg_sets[cond].shape[0]} images, expected {n_seg}.")

    # ---- Build defenses (constructs DiffPure model if enabled) ----
    defenses = build_defenses(args, device)

    # ---- Evaluation loops ----
    # results[defense][condition] = {"cls": acc, "miou": miou}
    results = {dname: {} for dname in defenses}

    bs = args.eval_batch_size

    for dname, dfunc in defenses.items():
        print(f"\n{'-' * 72}\nDEFENSE: {dname}\n{'-' * 72}")

        for cond in conditions:
            # ---- classification ----
            cls_imgs = cls_sets[cond]
            adv_correct, adv_total = 0, 0
            for s in tqdm(range(0, n_cls, bs),
                          desc=f"{dname}|{cond}|cls", leave=False):
                e = min(s + bs, n_cls)
                batch = cls_imgs[s:e].to(device)
                batch = dfunc(batch)                 # apply defense
                c, t = evaluate_classification_on_adversarial(
                    backbone, cls_head, batch, cls_labels[s:e], device
                )
                adv_correct += c
                adv_total += t
                del batch
                torch.cuda.empty_cache()
            cls_acc = 100.0 * adv_correct / adv_total

            # ---- segmentation ----
            seg_imgs = seg_sets[cond]
            conf = torch.zeros(21, 21, dtype=torch.long, device=device)
            for s in tqdm(range(0, n_seg, bs),
                          desc=f"{dname}|{cond}|seg", leave=False):
                e = min(s + bs, n_seg)
                batch = seg_imgs[s:e].to(device)
                batch = dfunc(batch)                 # apply defense
                conf += evaluate_segmentation_on_adversarial(
                    backbone, seg_head, batch, seg_masks[s:e], device
                )
                del batch
                torch.cuda.empty_cache()
            miou = confusion_to_miou(conf, num_classes=21)

            results[dname][cond] = {"cls": cls_acc, "miou": miou}
            print(f"  {cond:14s}  cls={cls_acc:6.2f}%   mIoU={miou:6.2f}%")

    # ============================================================
    # Tables
    # ============================================================
    print(f"\n{'=' * 72}\nRESULTS: Classification accuracy (%)\n{'=' * 72}")
    headers = ["Defense"] + conditions
    cls_rows = []
    for dname in defenses:
        row = [dname] + [f"{results[dname][c]['cls']:.2f}" for c in conditions]
        cls_rows.append(row)
    print(tabulate(cls_rows, headers=headers, tablefmt="grid"))

    print(f"\n{'=' * 72}\nRESULTS: Segmentation mIoU (%)\n{'=' * 72}")
    seg_rows = []
    for dname in defenses:
        row = [dname] + [f"{results[dname][c]['miou']:.2f}" for c in conditions]
        seg_rows.append(row)
    print(tabulate(seg_rows, headers=headers, tablefmt="grid"))

    # ---- Drop-vs-own-clean view (the key relative comparison) ----
    print(f"\n{'=' * 72}")
    print("ATTACK STRENGTH RETAINED: drop vs Clean (pp), per defense")
    print("(higher = attack still effective AFTER the defense)")
    print(f"{'=' * 72}")
    if "Clean" in conditions:
        drop_rows = []
        atk_conds = [c for c in conditions if c != "Clean"]
        for dname in defenses:
            clean_cls = results[dname]["Clean"]["cls"]
            clean_miou = results[dname]["Clean"]["miou"]
            row = [dname]
            for c in atk_conds:
                d_cls = clean_cls - results[dname][c]["cls"]
                d_miou = clean_miou - results[dname][c]["miou"]
                row.append(f"{d_cls:.1f}/{d_miou:.1f}")
            drop_rows.append(row)
        print(tabulate(drop_rows,
                       headers=["Defense"] + [f"{c} (cls/seg)" for c in atk_conds],
                       tablefmt="grid"))

    # ---- Save ----
    os.makedirs("./outputs/results", exist_ok=True)
    # Build filename suffix from --max_images and --tag.
    parts = []
    if args.max_images is not None:
        parts.append(f"n{args.max_images}")
    if args.tag:
        parts.append(args.tag)
    suffix = ("_" + "_".join(parts)) if parts else ""
    pt_path  = f"./outputs/results/defenses_voc_{args.model_name}{suffix}.pt"
    txt_path = f"./outputs/results/defenses_voc_{args.model_name}{suffix}.txt"
    torch.save(results, pt_path)
    with open(txt_path, "w") as f:
        f.write(f"PascalVOC defenses — {args.model_name}\n")
        f.write(f"n_cls={n_cls}, n_seg={n_seg}, "
                f"jpeg_qualities={args.jpeg_qualities}, "
                f"diffpure_tstar={args.diffpure_tstar}, "
                f"no_diffpure={args.no_diffpure}\n\n")
        f.write("Classification accuracy (%):\n")
        f.write(tabulate(cls_rows, headers=headers, tablefmt="grid"))
        f.write("\n\nSegmentation mIoU (%):\n")
        f.write(tabulate(seg_rows, headers=headers, tablefmt="grid"))
    print(f"\nSaved:\n  {txt_path}\n  {pt_path}")


if __name__ == "__main__":
    main()
