"""
Multi-label classification mAP for the ablation, defense and
cross-encoder experiments.

The ablation, defense and cross-encoder scripts evaluate classification
with the single-label head (top-1 accuracy). This script re-scores the
same cached adversarial images with the multi-label head and mAP
(src/metrics_multilabel.py), which is the classification metric reported
in the paper. No attack is run; segmentation is not affected.

Targets:
    table13         reference construction (fixed clean vs. transform-both), both backbones
    table14         transformation-family composition, ViT-S/14
    table15         application probability and parameter range, ViT-S/14
    table16         input-preprocessing defenses, ViT-B/14, full VOC val
    jpegsweep_cls   JPEG q in {90, 75, 50, 30, 10}, ViT-B/14 (Figure 5)
    cross_encoder   cross-encoder transfer ViT-S/14 <-> ViT-B/14

USAGE
    python scripts/rescore_multilabel_all.py --target table13 --pascalvoc_root <VOC2012>
    (same for table14, table15, table16, jpegsweep_cls, cross_encoder)

OUTPUTS
    outputs/results/rescore_multilabel_<target>.txt / .pt
"""

import os
import sys
import argparse

import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

from src.models import get_dinov2_model, get_model_info
from src.heads_multilabel import MultiLabelClassificationHead
from src.metrics_multilabel import collect_scores_and_labels_on_adversarial, compute_map
from src.datasets import PascalVOCClassification
from src.datasets_multilabel import PascalVOCClassificationMultiLabel
from src.defenses import build_defenses, DiffPure
from evaluate_eot_taa_clean import get_eval_transforms


# ============================================================
# Shared helpers
# ============================================================

def build_aligned_labels(pascalvoc_root, image_size, expected_N):
    """(N, 20) multi-hot labels aligned to PascalVOCClassification's image order."""
    sl_ds = PascalVOCClassification(root=pascalvoc_root, split="val", image_size=image_size)
    ml_ds = PascalVOCClassificationMultiLabel(root=pascalvoc_root, split="val", image_size=image_size)
    if len(ml_ds) != len(sl_ds):
        raise RuntimeError(f"Multi-label dataset has {len(ml_ds)} items but single-label has {len(sl_ds)}.")
    for i in [0, len(sl_ds) // 4, len(sl_ds) // 2, 3 * len(sl_ds) // 4, len(sl_ds) - 1]:
        if sl_ds.images[i] != ml_ds.images[i]:
            raise RuntimeError(f"Dataset alignment broken at index {i}.")
    labels = np.stack(ml_ds.labels, axis=0).astype(np.float32)
    labels_tensor = torch.from_numpy(labels)
    if labels_tensor.shape[0] < expected_N:
        raise RuntimeError(f"Multi-label dataset smaller than needed: {labels_tensor.shape[0]} < {expected_N}")
    return labels_tensor[:expected_N]


def load_backbone_and_head(model_name, heads_dir, device):
    info = get_model_info(model_name)
    backbone = get_dinov2_model(model_name, device)
    backbone.eval()
    head = MultiLabelClassificationHead(info["dim"], 20).to(device)
    path = os.path.join(heads_dir, f"{model_name}_classification_head_multilabel.pth")
    head.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    head.eval()
    return backbone, head


def true_map(backbone, head, images, labels, device, batch_size=16, transform_fn=None, desc=""):
    all_scores, all_labels = [], []
    for s in tqdm(range(0, images.shape[0], batch_size), desc=desc, leave=False):
        e = min(s + batch_size, images.shape[0])
        batch = images[s:e].to(device)
        if transform_fn is not None:
            batch = transform_fn(batch)
        sc, lb = collect_scores_and_labels_on_adversarial(backbone, head, batch, labels[s:e].to(device), device)
        all_scores.append(sc)
        all_labels.append(lb)
        torch.cuda.empty_cache()
    return compute_map(np.concatenate(all_scores, 0), np.concatenate(all_labels, 0))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def clean_map_by_transform(backbone, head, clean_images, labels, device, batch_size, desc=""):
    """mAP of the clean images under each of the 9+1 eval transforms. Shared
    across every attack condition, so compute once per (backbone, clean set)."""
    transforms_dict = get_eval_transforms()
    out = {}
    for t_name, (t_func, t_kwargs) in transforms_dict.items():
        tf = (lambda x, f=t_func, kw=t_kwargs: f(x, **kw))
        out[t_name] = true_map(backbone, head, clean_images, labels, device, batch_size,
                                transform_fn=tf, desc=f"{desc}/{t_name}/clean")
    return out


def avg_degradation_over_9(backbone, head, adv_images, clean_by_transform, labels, device, batch_size, desc=""):
    """Average (clean - adv) mAP over the 9 real transforms, plus the untransformed drop.
    Returns (avg_drop_9, untransf_drop, per_transform_dict)."""
    transforms_dict = get_eval_transforms()
    per_t = {}
    for t_name, (t_func, t_kwargs) in transforms_dict.items():
        tf = (lambda x, f=t_func, kw=t_kwargs: f(x, **kw))
        m_adv = true_map(backbone, head, adv_images, labels, device, batch_size,
                          transform_fn=tf, desc=f"{desc}/{t_name}/adv")
        per_t[t_name] = {"clean": clean_by_transform[t_name], "adv": m_adv,
                          "drop": clean_by_transform[t_name] - m_adv}
    avg_drop_9 = float(np.mean([v["drop"] for k, v in per_t.items() if k != "No transform"]))
    untransf_drop = per_t["No transform"]["drop"]
    return avg_drop_9, untransf_drop, per_t


def save(name, table_str, obj):
    os.makedirs("outputs/results", exist_ok=True)
    with open(f"outputs/results/rescore_multilabel_{name}.txt", "w") as f:
        f.write(table_str)
    torch.save(obj, f"outputs/results/rescore_multilabel_{name}.pt")
    print(f"\nSaved:\n  outputs/results/rescore_multilabel_{name}.txt\n  outputs/results/rescore_multilabel_{name}.pt")


# ============================================================
# Target: Table 13 (reference construction), both backbones
# ============================================================

def run_table13(args, device):
    rows = []
    result = {}
    for model_name in ["dinov2_vits14", "dinov2_vitb14"]:
        print(f"\n{'=' * 72}\nTable 13: {model_name}\n{'=' * 72}")
        backbone, head = load_backbone_and_head(model_name, args.heads_dir, device)
        cache = f"outputs/cached_images_ablation/{model_name}"
        labels_full = build_aligned_labels(args.pascalvoc_root, args.image_size, 500)

        clean = load(os.path.join(cache, "cls_clean.pt"))
        clean_bt = clean_map_by_transform(backbone, head, clean, labels_full[:clean.shape[0]],
                                          device, args.eval_batch_size, desc=model_name)

        for cond, fname in [("both", "cls_adv_both_K8.pt"), ("advonly", "cls_adv_advonly_K8.pt")]:
            imgs = load(os.path.join(cache, fname))
            n = imgs.shape[0]
            avg9, untransf, per_t = avg_degradation_over_9(
                backbone, head, imgs, clean_bt, labels_full[:n], device,
                args.eval_batch_size, desc=f"{model_name}/{cond}")
            print(f"  {cond}: avg-9={avg9:.2f}  untransf={untransf:.2f}")
            rows.append([model_name, cond, f"{avg9:.2f}", f"{untransf:.2f}"])
            result[f"{model_name}_{cond}"] = {"avg9": avg9, "untransf": untransf, "per_transform": per_t}
        del backbone, head
        torch.cuda.empty_cache()

    table = tabulate(rows, headers=["Backbone", "Condition", "Avg-9 drop", "Untransf. drop"], tablefmt="grid")
    print(table)
    save("table13", table, result)


# ============================================================
# Target: Table 14 (family), ViT-S/14 only
# ============================================================

def run_table14(args, device):
    model_name = "dinov2_vits14"
    print(f"\n{'=' * 72}\nTable 14: {model_name}\n{'=' * 72}")
    backbone, head = load_backbone_and_head(model_name, args.heads_dir, device)
    cache = f"outputs/cached_images_ablation_families/{model_name}"
    labels_full = build_aligned_labels(args.pascalvoc_root, args.image_size, 500)

    clean = load(os.path.join(cache, "cls_clean.pt"))
    clean_bt = clean_map_by_transform(backbone, head, clean, labels_full[:clean.shape[0]],
                                      device, args.eval_batch_size, desc=model_name)

    rows = []
    result = {}
    for cond in ["full", "degradation", "geometric", "photometric"]:
        imgs = load(os.path.join(cache, f"cls_adv_{cond}_K8.pt"))
        n = imgs.shape[0]
        avg9, untransf, per_t = avg_degradation_over_9(
            backbone, head, imgs, clean_bt, labels_full[:n], device, args.eval_batch_size, desc=cond)
        worst = min(v["drop"] for k, v in per_t.items() if k != "No transform")
        print(f"  {cond}: avg-9={avg9:.2f}  worst-case={worst:.2f}  untransf={untransf:.2f}")
        rows.append([cond, f"{avg9:.2f}", f"{worst:.2f}"])
        result[cond] = {"avg9": avg9, "worst": worst, "untransf": untransf, "per_transform": per_t}

    table = tabulate(rows, headers=["Family", "Avg-9 drop", "Worst-case"], tablefmt="grid")
    print(table)
    save("table14", table, result)


# ============================================================
# Target: Table 15 (probability + range), ViT-S/14 only
# ============================================================

def run_table15(args, device):
    model_name = "dinov2_vits14"
    print(f"\n{'=' * 72}\nTable 15: {model_name}\n{'=' * 72}")
    backbone, head = load_backbone_and_head(model_name, args.heads_dir, device)
    labels_full = build_aligned_labels(args.pascalvoc_root, args.image_size, 500)

    rows = []
    result = {}

    prob_cache = f"outputs/cached_images_ablation_prob/{model_name}"
    clean_p = load(os.path.join(prob_cache, "cls_clean.pt"))
    clean_bt_p = clean_map_by_transform(backbone, head, clean_p, labels_full[:clean_p.shape[0]],
                                        device, args.eval_batch_size, desc="prob-clean")
    for cond in ["halved", "published", "doubled"]:
        imgs = load(os.path.join(prob_cache, f"cls_adv_{cond}_K8.pt"))
        n = imgs.shape[0]
        avg9, untransf, per_t = avg_degradation_over_9(
            backbone, head, imgs, clean_bt_p, labels_full[:n], device, args.eval_batch_size, desc=f"prob/{cond}")
        print(f"  probability {cond}: avg-9={avg9:.2f}")
        rows.append([f"probability {cond}", f"{avg9:.2f}"])
        result[f"probability_{cond}"] = {"avg9": avg9, "untransf": untransf, "per_transform": per_t}

    range_cache = f"outputs/cached_images_ablation_range/{model_name}"
    clean_r = load(os.path.join(range_cache, "cls_clean.pt"))
    clean_bt_r = clean_map_by_transform(backbone, head, clean_r, labels_full[:clean_r.shape[0]],
                                        device, args.eval_batch_size, desc="range-clean")
    for cond in ["narrow", "published", "wide"]:
        imgs = load(os.path.join(range_cache, f"cls_adv_{cond}_K8.pt"))
        n = imgs.shape[0]
        avg9, untransf, per_t = avg_degradation_over_9(
            backbone, head, imgs, clean_bt_r, labels_full[:n], device, args.eval_batch_size, desc=f"range/{cond}")
        print(f"  range {cond}: avg-9={avg9:.2f}")
        rows.append([f"range {cond}", f"{avg9:.2f}"])
        result[f"range_{cond}"] = {"avg9": avg9, "untransf": untransf, "per_transform": per_t}

    table = tabulate(rows, headers=["Condition", "Avg-9 drop"], tablefmt="grid")
    print(table)
    save("table15", table, result)


# ============================================================
# Target: Table 16 (defenses), ViT-B/14, full dataset
# ============================================================

def run_table16(args, device):
    model_name = "dinov2_vitb14"
    print(f"\n{'=' * 72}\nTable 16: {model_name} (full dataset, 8 defenses x 4 conditions)\n{'=' * 72}")
    backbone, head = load_backbone_and_head(model_name, args.heads_dir, device)

    clean = load(f"outputs/cached_images_clean/{model_name}/cls_clean_images.pt")
    n = clean.shape[0]
    labels_full = build_aligned_labels(args.pascalvoc_root, args.image_size, n)

    cls_sets = {
        "Clean": clean,
        "TAA": load(f"outputs/cached_images/{model_name}/cls_adv_images.pt"),
        "EoT-TAA K=4": load(f"outputs/cached_images_clean/{model_name}/cls_adv_images_K4.pt"),
        "EoT-TAA K=8": load(f"outputs/cached_images_clean/{model_name}/cls_adv_images_K8.pt"),
    }

    class DummyArgs:
        jpeg_qualities = [75, 50, 10]
        bit_depth = 4
        no_resize_pad = False
        resize_pad_min_scale = 0.9
        no_median = False
        median_size = 3
        no_dither = False
        dither_levels = 4
        no_diffpure = True
        diffpure_tstar = 0.1
        diffpure_model = ""
    defenses = build_defenses(DummyArgs(), device)
    print(f"Defenses: {list(defenses.keys())}")

    rows = []
    result = {}
    for dname, dfunc in defenses.items():
        for cond, imgs in cls_sets.items():
            if cond == "Clean":
                continue
            m_adv = true_map(backbone, head, imgs, labels_full, device, args.eval_batch_size,
                             transform_fn=dfunc, desc=f"{dname}/{cond}")
            print(f"  {dname} | {cond}: mAP={m_adv:.2f}")
            rows.append([dname, cond, f"{m_adv:.2f}"])
            result[f"{dname}__{cond}"] = m_adv
        # clean reference under this defense too (needed for degradation later)
        m_clean = true_map(backbone, head, clean, labels_full, device, args.eval_batch_size,
                           transform_fn=dfunc, desc=f"{dname}/Clean")
        result[f"{dname}__Clean"] = m_clean
        rows.append([dname, "Clean", f"{m_clean:.2f}"])
        print(f"  {dname} | Clean: mAP={m_clean:.2f}")

    table = tabulate(rows, headers=["Defense", "Condition", "mAP"], tablefmt="grid")
    print(table)
    save("table16", table, result)


# ============================================================
# Target: JPEG sweep for Figure 5 (classification panel), ViT-B/14, full dataset
# ============================================================

def run_jpegsweep_cls(args, device):
    """Multi-label mAP under JPEG q in {90, 75, 50, 30, 10}. Table 16 only
    covers q in {75, 50, 10}; Figure 5 also needs q=90 and q=30. Same cached
    images and same key format as table16, so the two .pt files agree on the
    shared qualities."""
    model_name = "dinov2_vitb14"
    print(f"\n{'=' * 72}\nJPEG sweep (cls): {model_name}\n{'=' * 72}")
    backbone, head = load_backbone_and_head(model_name, args.heads_dir, device)

    clean = load(f"outputs/cached_images_clean/{model_name}/cls_clean_images.pt")
    n = clean.shape[0]
    labels_full = build_aligned_labels(args.pascalvoc_root, args.image_size, n)

    cls_sets = {
        "Clean": clean,
        "TAA": load(f"outputs/cached_images/{model_name}/cls_adv_images.pt"),
        "EoT-TAA K=8": load(f"outputs/cached_images_clean/{model_name}/cls_adv_images_K8.pt"),
    }

    class DummyArgs:
        jpeg_qualities = [90, 75, 50, 30, 10]
        no_bitdepth = True
        no_resize_pad = True
        no_median = True
        no_dither = True
        no_diffpure = True
    defenses = build_defenses(DummyArgs(), device)
    print(f"Defenses: {list(defenses.keys())}")

    rows = []
    result = {}
    for dname, dfunc in defenses.items():
        for cond, imgs in cls_sets.items():
            m = true_map(backbone, head, imgs, labels_full, device, args.eval_batch_size,
                         transform_fn=dfunc, desc=f"{dname}/{cond}")
            print(f"  {dname} | {cond}: mAP={m:.2f}")
            rows.append([dname, cond, f"{m:.2f}"])
            result[f"{dname}__{cond}"] = m

    table = tabulate(rows, headers=["Defense", "Condition", "mAP"], tablefmt="grid")
    print(table)
    save("jpegsweep_cls", table, result)


# ============================================================
# Target: Cross-encoder transfer, classification only
# ============================================================

def run_cross_encoder(args, device):
    MODELS = ["dinov2_vits14", "dinov2_vitb14"]
    CONDITIONS = [("TAA", "cached_images", "cls_adv_images.pt"),
                  ("EoT-TAA K=4", "cached_images_clean", "cls_adv_images_K4.pt"),
                  ("EoT-TAA K=8", "cached_images_clean", "cls_adv_images_K8.pt")]

    heads = {}
    backbones = {}
    for m in MODELS:
        backbones[m], heads[m] = load_backbone_and_head(m, args.heads_dir, device)

    clean = load(f"outputs/cached_images_clean/{MODELS[0]}/cls_clean_images.pt")
    n = clean.shape[0]
    labels_full = build_aligned_labels(args.pascalvoc_root, args.image_size, n)

    clean_ref = {}
    for target in MODELS:
        m = true_map(backbones[target], heads[target], clean, labels_full, device,
                     args.eval_batch_size, desc=f"clean/{target}")
        clean_ref[target] = m
        print(f"Clean mAP on {target}: {m:.2f}")

    rows = []
    result = {}
    for source in MODELS:
        for cond_name, cache_dir, cls_file in CONDITIONS:
            cls_path = os.path.join("outputs", cache_dir, source, cls_file)
            if not os.path.exists(cls_path):
                continue
            cls_adv = load(cls_path)
            for target in MODELS:
                mode = "white-box" if source == target else "black-box transfer"
                m_adv = true_map(backbones[target], heads[target], cls_adv, labels_full, device,
                                 args.eval_batch_size, desc=f"{source}->{target}/{cond_name}")
                drop = clean_ref[target] - m_adv
                print(f"{source} -> {target} ({mode}), {cond_name}: mAP {clean_ref[target]:.2f} -> {m_adv:.2f} (drop {drop:.2f})")
                rows.append([source, target, mode, cond_name, f"{drop:.2f}"])
                result[f"{source}_{target}_{cond_name}"] = {"clean": clean_ref[target], "adv": m_adv, "drop": drop}

    table = tabulate(rows, headers=["Source", "Target", "Mode", "Condition", "Cls drop (pp)"], tablefmt="grid")
    print(table)
    save("cross_encoder", table, result)


def run_diffpure_cls(args, device):
    """Classification (mAP) under DiffPure only, ViT-B/14, Clean/TAA/EoT-TAA K=8
    (not used in the paper). Full dataset unless --diffpure_max_images caps it."""
    model_name = "dinov2_vitb14"
    print(f"\n{'=' * 72}\nDiffPure classification: {model_name} "
          f"(Clean/TAA/EoT-TAA K=8, batch={args.diffpure_batch_size})\n{'=' * 72}")
    backbone, head = load_backbone_and_head(model_name, args.heads_dir, device)

    clean = load(f"outputs/cached_images_clean/{model_name}/cls_clean_images.pt")
    taa = load(f"outputs/cached_images/{model_name}/cls_adv_images.pt")
    k8 = load(f"outputs/cached_images_clean/{model_name}/cls_adv_images_K8.pt")

    n = clean.shape[0]
    if args.diffpure_max_images is not None:
        n = min(n, args.diffpure_max_images)
        clean, taa, k8 = clean[:n], taa[:n], k8[:n]
    print(f"n={n} images per condition")

    labels_full = build_aligned_labels(args.pascalvoc_root, args.image_size, n)

    dp = DiffPure(device=device, t_star=args.diffpure_tstar,
                  num_inference_steps=args.diffpure_steps,
                  model_id=(args.diffpure_model or None))
    dp_name = f"DiffPure t*={args.diffpure_tstar} ({dp.loaded_model_id.split('/')[-1]})"
    print(f"Defense: {dp_name}")

    import time
    cls_sets = {"Clean": clean, "TAA": taa, "EoT-TAA K=8": k8}
    result = {"defense": dp_name, "n": n}
    rows = []
    for cond, imgs in cls_sets.items():
        t0 = time.time()
        m = true_map(backbone, head, imgs, labels_full, device,
                     args.diffpure_batch_size, transform_fn=dp, desc=f"DiffPure/{cond}")
        dt = time.time() - t0
        print(f"  {cond}: mAP={m:.2f}  ({dt:.1f}s, {dt/n:.2f}s/img)")
        rows.append([cond, f"{m:.2f}", f"{dt:.1f}s"])
        result[cond] = m

    result["TAA_drop"] = result["Clean"] - result["TAA"]
    result["K8_drop"] = result["Clean"] - result["EoT-TAA K=8"]
    rows.append(["TAA drop (pp)", f"{result['TAA_drop']:.2f}", ""])
    rows.append(["K8 drop (pp)", f"{result['K8_drop']:.2f}", ""])

    table = tabulate(rows, headers=["Condition", "mAP", "time"], tablefmt="grid")
    print(table)
    save("diffpure_cls" + (f"_n{n}" if args.diffpure_max_images else ""), table, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["table13", "table14", "table15", "table16", "jpegsweep_cls",
                             "cross_encoder", "diffpure_cls"])
    ap.add_argument("--pascalvoc_root", type=str, required=True)
    ap.add_argument("--heads_dir", type=str, default="./checkpoints")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--diffpure_batch_size", type=int, default=8)
    ap.add_argument("--diffpure_tstar", type=float, default=0.1)
    ap.add_argument("--diffpure_steps", type=int, default=None)
    ap.add_argument("--diffpure_model", type=str, default="")
    ap.add_argument("--diffpure_max_images", type=int, default=None)
    args = ap.parse_args()

    device = args.device
    {
        "table13": run_table13,
        "table14": run_table14,
        "table15": run_table15,
        "table16": run_table16,
        "jpegsweep_cls": run_jpegsweep_cls,
        "cross_encoder": run_cross_encoder,
        "diffpure_cls": run_diffpure_cls,
    }[args.target](args, device)


if __name__ == "__main__":
    main()
