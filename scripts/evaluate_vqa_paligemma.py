"""
PaliGemma 2 VQA evaluation of cached clean / TAA / TR-TAA images.

Loads the images written by attack_paligemma_vision_tower.py, answers every
VQAv2 question of each image with PaliGemma 2 (google/paligemma2-3b-mix-448,
greedy decoding, prompt "answer en <question>"), and scores the answers with
the standard VQA accuracy (leave-one-annotator-out average of
min(#matches/3, 1)). Reports overall and per-answer-type accuracy and the
accuracy drop relative to clean images.

Evaluation modes (one per run):
    (default)      untransformed images only
    --transforms   the nine evaluation transformations + untransformed
    --defenses     JPEG q=50, bit-depth reduction, resize-and-pad, median filter
    --dither       Floyd-Steinberg dithering
    --jpeg_sweep   JPEG q in {75, 30, 10}

USAGE
    python scripts/evaluate_vqa_paligemma.py --vqav2_root <VQAv2> --transforms
    python scripts/evaluate_vqa_paligemma.py --vqav2_root <VQAv2> --defenses
    python scripts/evaluate_vqa_paligemma.py --vqav2_root <VQAv2> --dither

OUTPUTS
    ./outputs/results/vqa_paligemma_siglip_so400m{,_defenses,_dither,_jpegsweep}.{txt,pt}
"""

import os
import sys
import io
import argparse
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from tqdm import tqdm
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets_vqav2 import (
    VQAv2Subsample, load_questions_and_annotations,
)
from src.metrics_vqa import aggregate_vqa_scores
from src.defenses import bit_depth_reduction, randomized_resize_pad, median_filter_defense, dither_defense, DiffPure


# ============================================================
# Transforms (same suite and parameters as the other tasks)
# Operate on (B,3,H,W) tensors in [0,1].
# ============================================================

def apply_jpeg(images, quality=50):
    out = []
    for i in range(images.shape[0]):
        im = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(TF.to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(out).to(images.device)


def apply_blur(images, sigma=2.0):
    out = []
    for i in range(images.shape[0]):
        im = TF.to_pil_image(images[i].cpu().clamp(0, 1))
        out.append(TF.to_tensor(im.filter(ImageFilter.GaussianBlur(radius=sigma))))
    return torch.stack(out).to(images.device)


def apply_resize(images, scale=0.5):
    B, C, H, W = images.shape
    small = F.interpolate(images, scale_factor=scale, mode="bilinear",
                          align_corners=False)
    return F.interpolate(small, size=(H, W), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def apply_noise(images, sigma=0.03):
    return (images + torch.randn_like(images) * sigma).clamp(0, 1)


def apply_rotation(images, angle=15):
    """Rotate then center-crop to remove black corners, then resize back."""
    rot = TF.rotate(images, angle)
    H = images.shape[-1]
    c = int(H * 0.78)
    o = (H - c) // 2
    crop = rot[..., o:o + c, o:o + c]
    return F.interpolate(crop, size=(H, H), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def apply_pipeline(images):
    images = apply_resize(images, scale=0.5)
    images = apply_blur(images, sigma=1.5)
    images = apply_noise(images, sigma=0.02)
    images = apply_jpeg(images, quality=60)
    return images


def get_eval_transforms(full):
    if not full:
        return {"No transform": (lambda x: x, {})}
    return {
        "No transform":  (lambda x: x, {}),
        "JPEG q=50":     (apply_jpeg,     {"quality": 50}),
        "JPEG q=30":     (apply_jpeg,     {"quality": 30}),
        "Blur s=1.0":    (apply_blur,     {"sigma": 1.0}),
        "Blur s=2.0":    (apply_blur,     {"sigma": 2.0}),
        "Resize 50%":    (apply_resize,   {"scale": 0.5}),
        "Resize 25%":    (apply_resize,   {"scale": 0.25}),
        "Rotation 15":   (apply_rotation, {"angle": 15}),
        "Noise s=0.03":  (apply_noise,    {"sigma": 0.03}),
        "Print-scan":    (apply_pipeline, {}),
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
# PaliGemma generation
# ============================================================

@torch.no_grad()
def paligemma_answer_batch(model, processor, pil_images, questions, device,
                           max_new_tokens=10):
    """
    Batched PaliGemma 2 VQA generation.
    pil_images: list of PIL images
    questions: list of question strings (same length)
    Returns: list of decoded answer strings
    """
    prompts = [f"answer en {q}\n" for q in questions]
    inputs = processor(text=prompts, images=pil_images,
                       return_tensors="pt", padding="longest")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False)
    gen = out[:, input_len:]
    return [processor.decode(g, skip_special_tokens=True).strip() for g in gen]


def tensor01_to_pil(t):
    return TF.to_pil_image(t.clamp(0, 1).cpu())


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqav2_root", type=str, required=True)
    ap.add_argument("--cache_dir", type=str,
                    default="./outputs/cached_images_vqav2/victim_final")
    ap.add_argument("--paligemma_model", type=str,
                    default="google/paligemma2-3b-mix-448")
    ap.add_argument("--subsample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_images", type=int, default=None,
                    help="Cap images evaluated (for quick checks).")
    ap.add_argument("--question_batch", type=int, default=8,
                    help="Questions per PaliGemma generate() call.")
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--transforms", action="store_true",
                    help="Evaluate under the reduced transforms suite.")
    ap.add_argument("--defenses", action="store_true",
                    help="Evaluate under the 4-defense purification suite "
                         "(JPEG/BitDepth/ResizePad/Median) instead of the "
                         "10-transform robustness suite.")
    ap.add_argument("--jpeg_sweep", action="store_true",
                    help="Evaluate under JPEG q={75,30,10} only.")
    ap.add_argument("--dither", action="store_true",
                    help="Evaluate under multi-level Floyd-Steinberg "
                         "dithering only.")
    ap.add_argument("--diffpure", action="store_true",
                    help="Evaluate under diffusion-based purification "
                         "(DiffPure) only.")
    ap.add_argument("--diffpure_tstar", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device

    print(f"\n{'=' * 72}")
    print(f"PALIGEMMA 2 VQA EVALUATION (Clean / TAA / TR-TAA)")
    print(f"PaliGemma: {args.paligemma_model}")
    print(f"Cache: {args.cache_dir}")
    print(f"{'=' * 72}\n")

    # ---- Load cached image_ids order (defines image order in tensors) ----
    image_ids_path = os.path.join(args.cache_dir, "test_image_ids.pt")
    if not os.path.exists(image_ids_path):
        raise SystemExit(f"Missing {image_ids_path}; run generation first.")
    image_ids = torch.load(image_ids_path, weights_only=False)
    n_images_cached = len(image_ids)
    print(f"Cached image order: {n_images_cached} images")

    # ---- Build question lookup for those images ----
    print("Loading VQAv2 manifests for questions/answers...")
    questions_by_id, annotations_by_id, image_to_qids = (
        load_questions_and_annotations(args.vqav2_root)
    )

    # ---- Discover available attack conditions ----
    def cache_path(tag):
        return os.path.join(args.cache_dir, f"test_adv_images_{tag}.pt")

    conditions = {}
    clean_path = os.path.join(args.cache_dir, "test_clean_images.pt")
    if os.path.exists(clean_path):
        conditions["Clean"] = clean_path
    for tag, name in [("TAA", "TAA"), ("TRTAA", "TR-TAA"),
                      ("K4", "EoT-TAA K=4"), ("K8", "EoT-TAA K=8")]:
        if os.path.exists(cache_path(tag)):
            conditions[name] = cache_path(tag)

    print(f"Available conditions: {list(conditions.keys())}")
    if "Clean" not in conditions:
        raise SystemExit("Clean cache missing; cannot compute deltas.")

    n_eval = n_images_cached if args.max_images is None else min(
        args.max_images, n_images_cached)
    print(f"Evaluating on {n_eval} images.\n")

    # ---- Load PaliGemma 2 ----
    print(f"Loading PaliGemma 2 ({args.paligemma_model})...")
    from transformers import (
        AutoProcessor, PaliGemmaForConditionalGeneration,
    )
    processor = AutoProcessor.from_pretrained(args.paligemma_model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.paligemma_model, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"  loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    if args.diffpure:
        transforms_dict = get_diffpure_transforms(device, t_star=args.diffpure_tstar)
    elif args.dither:
        transforms_dict = get_dither_transforms()
    elif args.jpeg_sweep:
        transforms_dict = get_jpeg_sweep_transforms()
    elif args.defenses:
        transforms_dict = get_defense_transforms()
    else:
        transforms_dict = get_eval_transforms(args.transforms)
    baseline_t = next(iter(transforms_dict.keys()))  # "No transform" / "No defense"

    # all_results[cond][transform] = aggregate dict
    all_results = {}

    for cond_name, path in conditions.items():
        print(f"\n{'=' * 72}\nCONDITION: {cond_name}\n{'=' * 72}")
        imgs = torch.load(path, map_location="cpu", weights_only=False)
        # imgs: (N, 3, H, W) in [0, 1]
        res = imgs.shape[-1]
        all_results[cond_name] = {}

        for t_name, (t_func, t_kwargs) in transforms_dict.items():
            per_question = []

            for img_idx in tqdm(range(n_eval),
                                desc=f"{cond_name}|{t_name}", leave=False):
                img_id = image_ids[img_idx]
                qids = sorted(image_to_qids[img_id])

                # Build this image's questions + GT
                qs, gts, atypes = [], [], []
                for qid in qids:
                    q = questions_by_id[qid]["question"]
                    a = annotations_by_id[qid]
                    qs.append(q)
                    gts.append([ans["answer"] for ans in a["answers"]])
                    atypes.append(a["answer_type"])

                # Apply transform to the single image (add batch dim)
                x = imgs[img_idx:img_idx + 1]
                if t_func is not None:
                    x = t_func(x, **(t_kwargs or {}))
                pil = tensor01_to_pil(x[0])

                # Run questions in batches
                preds = []
                for s in range(0, len(qs), args.question_batch):
                    e = min(s + args.question_batch, len(qs))
                    batch_pils = [pil] * (e - s)
                    batch_qs = qs[s:e]
                    batch_preds = paligemma_answer_batch(
                        model, processor, batch_pils, batch_qs, device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    preds.extend(batch_preds)

                for pred, gt, atype in zip(preds, gts, atypes):
                    per_question.append({
                        "pred": pred, "gt_answers": gt, "answer_type": atype,
                    })

            agg = aggregate_vqa_scores(per_question, scorer="loo")
            all_results[cond_name][t_name] = agg
            print(f"  {t_name:14s}  overall={agg['overall']:.2f}%  "
                  f"yes/no={agg['yes/no']:.2f}%  number={agg['number']:.2f}%  "
                  f"other={agg['other']:.2f}%  (n={agg['n_total']})")

    # ============================================================
    # Tables
    # ============================================================
    print(f"\n{'=' * 72}")
    print(f"VQA ACCURACY (%) — '{baseline_t}'")
    print(f"{'=' * 72}")
    rows = []
    for cond in all_results:
        a = all_results[cond][baseline_t]
        rows.append([cond, f"{a['overall']:.2f}", f"{a['yes/no']:.2f}",
                     f"{a['number']:.2f}", f"{a['other']:.2f}"])
    print(tabulate(rows,
                   headers=["Condition", "Overall", "yes/no", "number", "other"],
                   tablefmt="grid"))

    base = all_results["Clean"][baseline_t]
    print(f"\n{'=' * 72}")
    print(f"Δ vs Clean (pp) — '{baseline_t}'  (more negative = stronger attack)")
    print(f"{'=' * 72}")
    drows = []
    for cond in all_results:
        if cond == "Clean":
            continue
        a = all_results[cond][baseline_t]
        drows.append([cond, f"{a['overall']-base['overall']:+.2f}",
                      f"{a['yes/no']-base['yes/no']:+.2f}",
                      f"{a['number']-base['number']:+.2f}",
                      f"{a['other']-base['other']:+.2f}"])
    print(tabulate(drows,
                   headers=["Condition", "ΔOverall", "Δyes/no", "Δnumber", "Δother"],
                   tablefmt="grid"))

    if args.transforms or args.defenses or args.jpeg_sweep or args.dither or args.diffpure:
        print(f"\n{'=' * 72}")
        print("VQA ACCURACY (%) under transforms — overall")
        print(f"{'=' * 72}")
        tnames = list(transforms_dict.keys())
        headers = ["Transform"] + list(all_results.keys())
        trows = []
        for t in tnames:
            row = [t]
            for c in all_results:
                row.append(f"{all_results[c][t]['overall']:.2f}")
            trows.append(row)
        print(tabulate(trows, headers=headers, tablefmt="grid"))

    # ---- Save ----
    os.makedirs("./outputs/results", exist_ok=True)
    suffix = "_diffpure" if args.diffpure else ("_dither" if args.dither else ("_jpegsweep" if args.jpeg_sweep else ("_defenses" if args.defenses else "")))
    pt_path  = f"./outputs/results/vqa_paligemma_siglip_so400m{suffix}.pt"
    txt_path = f"./outputs/results/vqa_paligemma_siglip_so400m{suffix}.txt"
    torch.save(all_results, pt_path)
    with open(txt_path, "w") as f:
        f.write(f"PaliGemma 2 VQA evaluation — {args.paligemma_model}\n")
        f.write(f"Attacks at {res}px; n_eval={n_eval} images\n\n")
        f.write(f"{baseline_t} accuracy:\n")
        f.write(tabulate(rows,
                headers=["Condition", "Overall", "yes/no", "number", "other"],
                tablefmt="grid"))
        f.write(f"\n\nDelta vs Clean ({baseline_t}):\n")
        f.write(tabulate(drows,
                headers=["Condition", "dOverall", "dyes/no", "dnumber", "dother"],
                tablefmt="grid"))
        if args.transforms or args.defenses or args.jpeg_sweep or args.dither or args.diffpure:
            f.write("\n\nAccuracy under transforms (overall):\n")
            f.write(tabulate(trows, headers=headers, tablefmt="grid"))
    print(f"\nSaved:\n  {txt_path}\n  {pt_path}")


if __name__ == "__main__":
    main()
