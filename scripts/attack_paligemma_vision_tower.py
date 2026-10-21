"""
TAA and TR-TAA against the vision tower of PaliGemma 2 (VQA experiments).

The vision tower is extracted from the PaliGemma 2 checkpoint
(google/paligemma2-3b-mix-448) and attacked at its native 448x448
resolution, on its patch-token output (the input of the multimodal
projector). The settings match the DINOv2 experiments: T=50 PGD steps,
eps=8/255, K=8, the transformation distribution of Table 1 and the fixed
clean reference. The step size defaults to 2.5*eps/T = 0.4/255. The
centering mean mu is estimated on the vision tower from 512 COCO val2014
images disjoint from the evaluation images.

Images: 500 COCO val2014 images (the VQAv2 validation images), drawn with
a fixed seed and resized to 448x448.

USAGE
    # TAA
    python scripts/attack_paligemma_vision_tower.py --vqav2_root <VQAv2> --attack taa

    # TR-TAA, K=8
    python scripts/attack_paligemma_vision_tower.py --vqav2_root <VQAv2> --attack eot --K 8

    # then evaluate with PaliGemma 2
    python scripts/evaluate_vqa_paligemma.py --vqav2_root <VQAv2> \
        --cache_dir ./outputs/cached_images_vqav2/victim_final [--transforms | --defenses | --dither]

OUTPUTS (in --out, default ./outputs/cached_images_vqav2/victim_final)
    test_adv_images_{TAA,K8}.pt   adversarial images (N, 3, 448, 448)
    test_clean_images.pt          clean images
    test_image_ids.pt             COCO image ids, in tensor order
    stats_{TAA,K8}.json           PSNR, victim-side cosine, attack settings
"""

import os
import sys
import json
import random
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eot_transforms_frozen import sample_random_transform


# ============================================================
# Victim encoder access
# ============================================================

def load_victim_tower(model_id, device, dtype=torch.float32):
    """
    Extract the vision tower from the PaliGemma checkpoint.

    Returns (tower, image_size, hidden_dim, processor). The tower is in eval
    mode with parameter gradients disabled; gradients are taken with respect
    to the input only. Attacks are crafted in float32.
    """
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

    full = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype,
    )
    # attribute location differs between transformers versions
    tower = getattr(full, "vision_tower", None)
    if tower is None:
        tower = full.model.vision_tower
    tower = tower.to(device).eval()
    for p in tower.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(model_id)
    ip = processor.image_processor
    size = ip.size
    image_size = size.get("height") or size.get("shortest_edge")

    cfg = tower.config
    vcfg = getattr(cfg, "vision_config", cfg)
    hidden = vcfg.hidden_size

    # the language model is not needed
    del full
    torch.cuda.empty_cache()

    return tower, int(image_size), int(hidden), processor


def siglip_normalize(x):
    """PaliGemma/SigLIP normalization: mean=std=0.5 on [0,1] input."""
    return (x - 0.5) / 0.5


def tower_patches(tower, x01):
    """
    x01: (B,3,H,W) in [0,1] at the tower's native resolution.
    Returns (B, N, D) patch tokens, i.e. the input of the multimodal projector.
    """
    out = tower(pixel_values=siglip_normalize(x01))
    return out.last_hidden_state


# ============================================================
# Deployment preprocessing (differentiable)
# ============================================================

def deployment_resample(x01, target):
    """
    Differentiable counterpart of the processor's bicubic resize.
    antialias=True matches PIL on downsampling; it is a no-op on upsampling.
    """
    if x01.shape[-1] == target and x01.shape[-2] == target:
        return x01
    return F.interpolate(
        x01, size=(target, target), mode="bicubic",
        align_corners=False, antialias=True,
    ).clamp(0, 1)


# ============================================================
# Feature loss
# ============================================================

def feature_cos(z_adv, z_o, mu, mode="flat"):
    """
    Cosine similarity between mean-centred patch tokens.
    "flat": one cosine over the flattened N*D vector (as in src/eot_attack.py;
            used for all results). "patch": mean of per-patch cosines.
    """
    mu_b = mu.view(1, 1, -1)
    a = z_adv - mu_b
    b = z_o - mu_b
    if mode == "flat":
        return F.cosine_similarity(
            a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=-1
        ).mean()
    an = F.normalize(a, p=2, dim=-1)
    bn = F.normalize(b, p=2, dim=-1)
    return (an * bn).sum(dim=-1).mean()


def quantize_uint8(x):
    """8-bit quantization, as at the end of src/eot_attack.py."""
    return ((x.clamp(0, 1) * 255.0).round() / 255.0)


# ============================================================
# Attacks against the victim tower
# ============================================================

def taa_victim(x01, tower, mu, native, eps_inf, step_size, num_steps,
               craft_size=None, loss_mode="flat"):
    """
    TAA on the victim tower.

    craft_size: resolution of the perturbation. Default = native.
    """
    craft_size = craft_size or x01.shape[-1]
    if x01.shape[-1] != craft_size:
        x01 = F.interpolate(x01, size=(craft_size, craft_size),
                            mode="bicubic", align_corners=False,
                            antialias=True).clamp(0, 1)

    x_adv = x01.clone().detach()

    # clean reference, computed once
    with torch.no_grad():
        z_o = tower_patches(tower, deployment_resample(x01, native))

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        z = tower_patches(tower, deployment_resample(x_adv, native))
        loss = feature_cos(z, z_o, mu, mode=loss_mode)

        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() - step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x01 + eps_inf), x01 - eps_inf)
        x_adv = x_adv.clamp(0, 1)

    return quantize_uint8(x_adv.detach())


def eot_taa_victim(x01, tower, mu, native, eps_inf, step_size, num_steps, K,
                   craft_size=None, deployment_in_eot=True,
                   transform_reference=False, loss_mode="flat"):
    """
    TR-TAA on the victim tower.

    transform_reference:
        False (default) -- the sampled transform t_k is applied to x_adv
            only; the reference is the fixed clean feature, computed once
            from the untransformed image (TR-TAA).
        True -- transform-both variant (ablation only).

    deployment_in_eot: append the victim's own resample to every sampled
        transformation.
    """
    craft_size = craft_size or x01.shape[-1]
    if x01.shape[-1] != craft_size:
        x01 = F.interpolate(x01, size=(craft_size, craft_size),
                            mode="bicubic", align_corners=False,
                            antialias=True).clamp(0, 1)

    x_adv = x01.clone().detach()

    dep = native if deployment_in_eot else None

    # fixed clean reference, computed once from the untransformed image
    z_o_fixed = None
    if not transform_reference:
        with torch.no_grad():
            z_o_fixed = tower_patches(tower, deployment_resample(x01, native))

    for _ in range(num_steps):
        total_grad = torch.zeros_like(x_adv)

        for _ in range(K):
            t_k = sample_random_transform(deployment_resize=dep)

            xk = x_adv.clone().detach().requires_grad_(True)
            x_adv_t = t_k(xk)

            z = tower_patches(tower, deployment_resample(x_adv_t, native))

            if transform_reference:
                with torch.no_grad():
                    x_o_t = t_k(x01)
                    z_o = tower_patches(
                        tower, deployment_resample(x_o_t, native))
            else:
                z_o = z_o_fixed

            loss = feature_cos(z, z_o, mu, mode=loss_mode)

            total_grad += torch.autograd.grad(loss, xk)[0].detach() / K

        x_adv = x_adv.detach() - step_size * total_grad.sign()
        x_adv = torch.max(torch.min(x_adv, x01 + eps_inf), x01 - eps_inf)
        x_adv = x_adv.clamp(0, 1)

    return quantize_uint8(x_adv.detach())


# ============================================================
# Metrics
# ============================================================

def per_image_psnr(clean, adv, quantize=True):
    """Per-image PSNR in dB (after 8-bit quantization), averaged over the batch."""
    if quantize:
        adv = (adv.clamp(0, 1) * 255).round() / 255.0
        clean = (clean.clamp(0, 1) * 255).round() / 255.0
    mse = ((clean - adv) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


@torch.no_grad()
def victim_cosine(tower, clean01, adv01, native, mu):
    """Mean per-patch cosine between clean and adversarial tokens at the
    victim encoder, after 8-bit quantization and the deployment resample."""
    q = lambda t: ((t.clamp(0, 1) * 255).round() / 255.0)
    mu_b = mu.view(1, 1, -1)
    zc = tower_patches(tower, deployment_resample(q(clean01), native))
    za = tower_patches(tower, deployment_resample(q(adv01), native))
    zc = F.normalize(zc - mu_b, p=2, dim=-1)
    za = F.normalize(za - mu_b, p=2, dim=-1)
    return (zc * za).sum(dim=-1).mean().item()


# ============================================================
# mu on the victim tower
# ============================================================

@torch.no_grad()
def compute_mu(tower, images_iter, native, device, n=512):
    """Empirical mean patch-token feature for centering, on the victim tower."""
    acc, count = None, 0
    for batch in tqdm(images_iter, desc="mu"):
        if count >= n:
            break
        x = batch.to(device)
        z = tower_patches(tower, deployment_resample(x, native))
        s = z.mean(dim=1).sum(dim=0)
        acc = s if acc is None else acc + s
        count += x.shape[0]
    return (acc / count).to(device)


# ============================================================
# Data
# ============================================================

def coco_files(vqav2_root, n_images, seed, exclude=None):
    d = os.path.join(vqav2_root, "val2014")
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    if exclude:
        files = [f for f in files if f not in exclude]
    random.Random(seed).shuffle(files)
    return [os.path.join(d, f) for f in files[:n_images]]


def coco_image_id(path):
    """COCO_val2014_000000395849.jpg -> 395849"""
    return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])


def load_batch(paths, size, device):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        ims.append(TF.to_tensor(im))
    return torch.stack(ims).to(device)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqav2_root", required=True)
    ap.add_argument("--model_id", default="google/paligemma2-3b-mix-448")
    ap.add_argument("--out", default="./outputs/cached_images_vqav2/victim_final")
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mu_images", type=int, default=512)

    ap.add_argument("--attack", choices=["taa", "eot"], default="taa",
                    help="'taa' = TAA, 'eot' = TR-TAA")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--craft_size", type=int, default=None,
                    help="Perturbation resolution. Default = victim native (448).")
    ap.add_argument("--eps_inf", type=float, default=8.0 / 255.0)
    ap.add_argument("--step_size", type=float, default=None,
                    help="Default = 2.5 * eps / num_steps (= 0.4/255).")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--no_deployment_eot", action="store_true")
    ap.add_argument("--loss", choices=["flat", "patch"], default="flat",
                    help="'flat' = one cosine over the flattened N*D vector "
                         "(as in src/eot_attack.py). 'patch' = mean of "
                         "per-patch cosines.")
    ap.add_argument("--transform_both", action="store_true",
                    help="Ablation only. Default: t_k is applied to x_adv and "
                         "the reference is the fixed clean feature.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device

    print("Loading PaliGemma vision tower...")
    tower, native, hidden, _ = load_victim_tower(args.model_id, device)
    print(f"  victim encoder: {native}x{native}, hidden={hidden}, "
          f"{(native // 14) ** 2} patch tokens")

    eps = args.eps_inf
    step = args.step_size or (2.5 * eps / args.num_steps)
    print(f"  eps={eps:.5f}  step={step:.6f}  steps={args.num_steps}")

    # ---- mu on the victim tower (different seed from the attacked images) ----
    mu_paths = coco_files(args.vqav2_root, args.mu_images, seed=43)
    def mu_iter():
        for i in range(0, len(mu_paths), args.batch_size):
            yield load_batch(mu_paths[i:i + args.batch_size], native, device)
    mu = compute_mu(tower, mu_iter(), native, device, n=args.mu_images)
    print(f"  mu: shape={tuple(mu.shape)} norm={mu.norm().item():.4f}")

    # ---- craft ----
    os.makedirs(args.out, exist_ok=True)
    paths = coco_files(args.vqav2_root, args.n_images, seed=args.seed)
    tag = "TAA" if args.attack == "taa" else f"K{args.K}"

    clean_all, adv_all, psnrs, coss = [], [], [], []
    for i in tqdm(range(0, len(paths), args.batch_size), desc=f"craft {tag}"):
        x = load_batch(paths[i:i + args.batch_size], native, device)

        if args.attack == "taa":
            a = taa_victim(x, tower, mu, native, eps, step, args.num_steps,
                           craft_size=args.craft_size, loss_mode=args.loss)
        else:
            a = eot_taa_victim(x, tower, mu, native, eps, step, args.num_steps,
                               K=args.K, craft_size=args.craft_size,
                               deployment_in_eot=not args.no_deployment_eot,
                               transform_reference=args.transform_both,
                               loss_mode=args.loss)

        if a.shape[-1] != x.shape[-1]:
            a = deployment_resample(a, x.shape[-1])

        clean_all.append(x.cpu())
        adv_all.append(a.cpu())
        psnrs.append(per_image_psnr(x, a))
        coss.append(victim_cosine(tower, x, a, native, mu))

    clean_all = torch.cat(clean_all)
    adv_all = torch.cat(adv_all)

    torch.save(adv_all, os.path.join(args.out, f"test_adv_images_{tag}.pt"))
    cp = os.path.join(args.out, "test_clean_images.pt")
    if not os.path.exists(cp):
        torch.save(clean_all, cp)
    ip = os.path.join(args.out, "test_image_ids.pt")
    if not os.path.exists(ip):
        torch.save([coco_image_id(p) for p in paths], ip)
    with open(os.path.join(args.out, f"stats_{tag}.json"), "w") as f:
        json.dump({
            "psnr_db": sum(psnrs) / len(psnrs),
            "victim_cosine": sum(coss) / len(coss),
            "eps_inf": eps, "step_size": step, "num_steps": args.num_steps,
            "craft_size": args.craft_size or native, "native": native,
            "attack": args.attack,
            "K": args.K if args.attack == "eot" else 1,
            "loss_mode": args.loss,
            "transform_reference": "both" if args.transform_both else "adv_only",
            "deployment_in_eot": not args.no_deployment_eot,
        }, f, indent=2)

    print(f"\n{tag}: PSNR={sum(psnrs) / len(psnrs):.2f} dB  "
          f"victim cosine={sum(coss) / len(coss):.4f}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
