"""
Image retrieval on Revisited Oxford (ROxford5k): generate TAA / TR-TAA
database images and evaluate retrieval under the nine evaluation
transformations and the input-preprocessing defenses.

Threat model: the 70 queries are clean and untransformed; the 4993
database images are adversarial and then pass through the transformation.
Each database image is attacked once per scale, and descriptors are then
extracted under every transformation from that same adversarial image.

Descriptor: DINOv2 CLS token, aspect-ratio-preserving resize (longer side
518), multi-scale average over scales {1, sqrt(2), 1/sqrt(2)}, L2
normalisation, no whitening. Metric: mAP under the Medium protocol;
degradation = mAP(t(clean db)) - mAP(t(adv db)).

Conditions:
    clean   no attack (run first; provides the transformed clean reference)
    taa     TAA (src.attacks.task_agnostic_attack)
    k4, k8  TR-TAA with K=4 / K=8 and the fixed clean reference

The centering mean mu is the PASCAL VOC mean of the same backbone
(outputs/cached_images_clean/<model_name>/mu_patch.pt).

USAGE
    for c in clean taa k8 k4; do
      python scripts/evaluate_roxford_transforms.py --roxford_root <ROxford> \
          --model_name dinov2_vits14 --condition $c
    done
    # defense study (single scale, ViT-B/14)
    for c in clean taa k8; do
      python scripts/evaluate_roxford_transforms.py --roxford_root <ROxford> \
          --model_name dinov2_vitb14 --condition $c --scales 1.0 --tag defenses
    done

OUTPUTS (results of every condition are merged into one file)
    outputs/results/retrieval_transforms_<model_name>[_<tag>].json
"""

import os
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import io
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from tqdm import tqdm

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT); sys.path.insert(0, _THIS)

from src.models import get_dinov2_model, get_model_info, normalize_for_model
from src.eot_attack import eot_taa_attack
from src.datasets_roxford import load_roxford_gnd
from src.metrics_retrieval_ref import compute_map, rank_database
from src.retrieval_preprocess import (
    load_image_aspect, pool_descriptor, fit_pcaw, apply_pcaw, _round14,
)
from src.datasets_roxford import build_difficulty_gnd
from src.defenses import bit_depth_reduction, randomized_resize_pad, median_filter_defense, dither_defense, DiffPure

# TAA -- the same function generate_taa_baseline.py uses for
# classification and segmentation.
from src.attacks import task_agnostic_attack as _plain_taa


# ============================================================
# Transforms -- must preserve (H, W), which is NOT square here
# because pad='none' keeps the aspect ratio.
# ============================================================

def t_identity(x):
    return x


def t_jpeg(x, quality):
    out = []
    for i in range(x.shape[0]):
        im = TF.to_pil_image(x[i].cpu().clamp(0, 1))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(TF.to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(out).to(x.device)


def t_blur(x, sigma):
    out = []
    for i in range(x.shape[0]):
        im = TF.to_pil_image(x[i].cpu().clamp(0, 1))
        out.append(TF.to_tensor(im.filter(ImageFilter.GaussianBlur(radius=sigma))))
    return torch.stack(out).to(x.device)


def t_resize(x, scale):
    H, W = x.shape[-2], x.shape[-1]
    h, w = max(14, int(H * scale)), max(14, int(W * scale))
    small = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(H, W), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def t_noise(x, sigma):
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def t_rotation(x, angle=15):
    """Rotate, centre-crop away the black corners, resize back to (H, W)."""
    H, W = x.shape[-2], x.shape[-1]
    rot = TF.rotate(x, angle)
    ch, cw = int(H * 0.78), int(W * 0.78)
    oh, ow = (H - ch) // 2, (W - cw) // 2
    crop = rot[..., oh:oh + ch, ow:ow + cw]
    return F.interpolate(crop, size=(H, W), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def t_printscan(x):
    x = t_resize(x, 0.5)
    x = t_blur(x, 1.5)
    x = t_noise(x, 0.02)
    return t_jpeg(x, 60)


TRANSFORMS = {
    "None":          lambda x: t_identity(x),
    "JPEG q=50":     lambda x: t_jpeg(x, 50),
    "JPEG q=30":     lambda x: t_jpeg(x, 30),
    "Blur s=1.0":    lambda x: t_blur(x, 1.0),
    "Blur s=2.0":    lambda x: t_blur(x, 2.0),
    "Resize 50%":    lambda x: t_resize(x, 0.5),
    "Resize 25%":    lambda x: t_resize(x, 0.25),
    "Rotation 15":   lambda x: t_rotation(x, 15),
    "Noise s=0.03":  lambda x: t_noise(x, 0.03),
    "Print-scan":    lambda x: t_printscan(x),
    "BitDepth b=4":  lambda x: bit_depth_reduction(x, bits=4),
    "ResizePad":     lambda x: randomized_resize_pad(x, min_scale=0.9),
    "Median k=3":    lambda x: median_filter_defense(x, size=3),
    "DitherFS L=4":  lambda x: dither_defense(x, levels=4),
}


# ============================================================
# Descriptor extraction, all transforms in one pass
# ============================================================

def extract_all_transforms(backbone, entries, image_size, scales, descriptor,
                           p, device, attack_fn, desc):
    """
    Returns {transform_name: (N, D) float32 numpy}, multi-scale averaged and
    L2-normalized.

    attack_fn: callable(x) -> x_adv, or None for the clean condition.
    """
    names = list(TRANSFORMS.keys())
    per_scale = {n: [] for n in names}

    for s in scales:
        size = _round14(image_size * s)
        feats = {n: [] for n in names}

        for (path, bbox) in tqdm(entries, desc=f"{desc}|scale{s:.2f}",
                                 leave=False):
            img = load_image_aspect(path, size, bbox=bbox, pad="none")
            x = img.unsqueeze(0).to(device)

            # attack ONCE, reuse for every transform
            if attack_fn is not None:
                x = attack_fn(x)
            x = x.detach()

            for n in names:
                xt = TRANSFORMS[n](x)
                with torch.no_grad():
                    out = backbone.forward_features(normalize_for_model(xt))
                feats[n].append(pool_descriptor(out, descriptor, p).cpu())

        for n in names:
            per_scale[n].append(torch.cat(feats[n], dim=0))

    result = {}
    for n in names:
        f = torch.stack(per_scale[n], dim=0).mean(dim=0)
        result[n] = torch.nn.functional.normalize(f, p=2, dim=1).numpy()
    return result


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roxford_root", required=True)
    ap.add_argument("--model_name", default="dinov2_vits14")
    ap.add_argument("--condition", required=True,
                    choices=["clean", "taa", "eot1", "k4", "k8"])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 1.4142, 0.7071])
    ap.add_argument("--descriptor", default="cls",
                    choices=["cls", "mean", "gem", "cls_patch"])
    ap.add_argument("--gem_p", type=float, default=3.0)
    ap.add_argument("--whiten", default="none", choices=["pcaw", "none"])
    ap.add_argument("--pad", default="none", choices=["none", "square"])
    ap.add_argument("--eps_inf", type=float, default=8.0 / 255.0)
    ap.add_argument("--step_size", type=float, default=0.0004)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--mu_path", default=None)
    ap.add_argument("--tag", default="",
                    help="Suffix for the output JSON filename, so a "
                         "different-scale/purpose run (e.g. the defense "
                         "study at --scales 1.0) doesn't overwrite the "
                         "main multi-scale results file.")
    ap.add_argument("--max_db_images", type=int, default=None,
                    help="Truncate the database to the first N images "
                         "(e.g. for an expensive defense like DiffPure). "
                         "Queries are unaffected.")
    ap.add_argument("--diffpure", action="store_true",
                    help="Evaluate under DiffPure only (None + DiffPure), "
                         "instead of the full TRANSFORMS dict, so this run "
                         "doesn't recompute the other defenses.")
    ap.add_argument("--diffpure_tstar", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.diffpure:
        dp = DiffPure(device=args.device, t_star=args.diffpure_tstar)
        dp_name = f"DiffPure t*={args.diffpure_tstar} ({dp.loaded_model_id.split('/')[-1]})"
        TRANSFORMS.clear()
        TRANSFORMS["None"] = lambda x: t_identity(x)
        TRANSFORMS[dp_name] = lambda x: dp(x)

    # fixed clean reference, as for every other task
    os.environ["EOT_TRANSFORM_REFERENCE"] = "0"

    backbone = get_dinov2_model(args.model_name, args.device).eval()
    cfg = load_roxford_gnd(args.roxford_root)
    diff = build_difficulty_gnd(cfg)

    mu_path = args.mu_path or os.path.join(
        _ROOT, "outputs/cached_images_clean", args.model_name, "mu_patch.pt")
    mu = torch.load(mu_path, map_location=args.device, weights_only=False)

    # ---- build the attack callable ----
    if args.condition == "clean":
        attack_fn = None
    elif args.condition == "taa":
        # TAA without transformation sampling (eot_taa_attack with K=1 would
        # still sample one transformation per step), as for the other tasks.
        attack_fn = lambda x: _plain_taa(
            x, backbone, mu, feature_type="patch", eps_inf=args.eps_inf,
            step_size=args.step_size, num_steps=args.num_steps,
            device=args.device)
    else:
        K = {"eot1": 1, "k4": 4, "k8": 8}[args.condition]
        attack_fn = lambda x: eot_taa_attack(
            x, backbone, mu, feature_type="patch", eps_inf=args.eps_inf,
            step_size=args.step_size, num_steps=args.num_steps, K=K,
            device=args.device)

    img_dir = cfg["image_dir"]
    q_entries = [(os.path.join(img_dir, cfg["qimlist"][i] + ".jpg"),
                  cfg["gnd"][i]["bbx"]) for i in range(cfg["n_q"])]
    db_entries = [(os.path.join(img_dir, nm + ".jpg"), None)
                  for nm in cfg["imlist"]]
    if args.max_db_images is not None:
        # Keep the first N in original order: ground-truth "ok"/"junk"
        # indices still refer to the same images at the same positions,
        # so compute_map stays correct -- ground-truth positives beyond N
        # simply become unretrievable, shrinking the candidate pool. This
        # makes absolute mAP not comparable to the full-database run, but
        # keeps the TAA-vs-TR-TAA comparison internally valid (same fixed
        # subset for every condition).
        db_entries = db_entries[:args.max_db_images]

    print(f"\n{args.model_name} | condition={args.condition} | "
          f"scales={args.scales} | {len(TRANSFORMS)} transforms | adv-only")
    print(f"R-Oxford: {cfg['n_db']} db, {cfg['n_q']} queries"
          f"{f' (truncated to {len(db_entries)} db images)' if args.max_db_images else ''}\n")

    # queries: clean AND untransformed (threat model: only the indexed
    # database passes through the deployment pipeline)
    q_only = {"None": None}
    saved = dict(TRANSFORMS)
    TRANSFORMS.clear(); TRANSFORMS["None"] = saved["None"]
    q_feats_all = extract_all_transforms(
        backbone, q_entries, args.image_size, args.scales, args.descriptor,
        args.gem_p, args.device, None, "queries")
    TRANSFORMS.clear(); TRANSFORMS.update(saved)
    q_raw = q_feats_all["None"]

    db_all = extract_all_transforms(
        backbone, db_entries, args.image_size, args.scales, args.descriptor,
        args.gem_p, args.device, attack_fn, f"DB({args.condition})")

    # ---- score every transform ----
    results = {}
    for tname, db_raw in db_all.items():
        if args.whiten == "pcaw":
            w_mean, w_P = fit_pcaw(db_raw)
            q_f = apply_pcaw(q_raw, w_mean, w_P)
            db_f = apply_pcaw(db_raw, w_mean, w_P)
        else:
            q_f, db_f = q_raw, db_raw

        ranks = rank_database(q_f, db_f)
        r = {}
        for proto in ["easy", "medium", "hard"]:
            m, _ = compute_map(ranks, diff[proto])
            r[proto] = float(m * 100.0)
        results[tname] = r
        print(f"  {tname:14s}  E={r['easy']:.1f}  M={r['medium']:.1f}  "
              f"H={r['hard']:.1f}")

    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(_ROOT, "outputs/results",
                       f"retrieval_transforms_{args.model_name}{suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    blob = {}
    if os.path.exists(out):
        with open(out) as f:
            blob = json.load(f)
    blob[args.condition] = {
        "results": results,
        "scales": args.scales,
        "descriptor": args.descriptor,
        "whiten": args.whiten,
        "eps_inf": args.eps_inf,
        "step_size": args.step_size,
        "num_steps": args.num_steps,
    }
    with open(out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nsaved -> {out}")

    if "clean" in blob and args.condition != "clean":
        base = blob["clean"]["results"]
        print(f"\nDelta vs transformed clean (Medium mAP, pp):")
        vals = []
        for tname in results:
            d = base[tname]["medium"] - results[tname]["medium"]
            print(f"  {tname:14s}  {d:6.1f}")
            if tname != "None":
                vals.append(d)
        print(f"  {'AVERAGE':14s}  {np.mean(vals):6.1f}  (excludes None)")


if __name__ == "__main__":
    main()
