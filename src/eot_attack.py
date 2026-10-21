"""
TR-TAA: Transformation-Robust Task-Agnostic Attack (Algorithm 1).

TR-TAA minimises the cosine similarity between the mean-centred patch-token
representation of a transformed adversarial image and that of the clean
image, in expectation over a distribution T of differentiable image
transformations:

    min_delta  E_{t~T} [ cos( f(t(x + delta)) - mu,  f(x) - mu ) ]
    s.t.       ||delta||_inf <= eps,  x + delta in [0, 1]

At every PGD step, K transformations are sampled independently, the
gradient is computed through each transformed view and averaged, and a
signed descent step is taken, followed by projection onto the eps-ball and
the valid image range. The clean reference f(x) - mu is computed once and
kept fixed. The final image is quantised to 8 bits.

The transformation distribution (Table 1 of the paper) consists of eight
differentiable operators -- Gaussian blur, down/up resizing, additive
Gaussian noise, brightness, contrast, horizontal flip, rotation and a
differentiable JPEG approximation -- each applied independently with its
own probability, in that order.

In file names and result keys of this repository, TR-TAA appears under its
working name "EoT-TAA" (e.g. "EoT-TAA K=8" is TR-TAA with K=8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import os

from src.models import normalize_for_model, VFMFeatureExtractor


# ============================================================
# Differentiable transforms for EoT
# All operate on GPU tensors (B, C, H, W) and allow gradient flow
# ============================================================

def diff_gaussian_blur(images, sigma=None):
    """Differentiable Gaussian blur using conv2d."""
    if sigma is None:
        sigma = random.uniform(0.5, 3.0)
    kernel_size = int(2 * round(3 * sigma) + 1)  # 6*sigma rule
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 21))

    # Create Gaussian kernel
    k = torch.arange(kernel_size, device=images.device, dtype=torch.float32)
    k = k - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (k / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    # Shape for depthwise conv: (C_out, 1, kH, kW) with groups=3
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    pad = kernel_size // 2
    padded = F.pad(images, [pad] * 4, mode='reflect')
    return F.conv2d(padded, gauss_2d, groups=3)


def diff_resize(images, scale=None):
    """Differentiable resize down then up (simulates resolution loss)."""
    if scale is None:
        scale = random.uniform(0.25, 0.75)
    B, C, H, W = images.shape
    small_h, small_w = max(4, int(H * scale)), max(4, int(W * scale))
    small = F.interpolate(images, size=(small_h, small_w), mode='bilinear',
                          align_corners=False)
    restored = F.interpolate(small, size=(H, W), mode='bilinear',
                             align_corners=False)
    return restored


def diff_noise(images, sigma=None):
    """Add Gaussian noise (differentiable — noise is constant wrt input)."""
    if sigma is None:
        sigma = random.uniform(0.01, 0.05)
    noise = torch.randn_like(images) * sigma
    return (images + noise).clamp(0, 1)


def diff_brightness(images, factor=None):
    """Differentiable brightness adjustment."""
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return (images * factor).clamp(0, 1)


def diff_contrast(images, factor=None):
    """Differentiable contrast adjustment."""
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    mean = images.mean(dim=(-2, -1), keepdim=True)
    return ((images - mean) * factor + mean).clamp(0, 1)


def diff_hflip(images):
    """Differentiable horizontal flip."""
    return images.flip(-1)


def diff_rotation(images, angle=None):
    """Differentiable rotation using affine grid."""
    if angle is None:
        angle = random.uniform(-30, 30)
    angle_rad = angle * np.pi / 180
    B = images.shape[0]

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation matrix
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=images.dtype, device=images.device).unsqueeze(0).expand(B, -1, -1)

    grid = F.affine_grid(theta, images.shape, align_corners=False)
    rotated = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros',
                            align_corners=False)
    return rotated


def diff_jpeg_approx(images, quality=None):
    """
    Approximate differentiable JPEG via blur + noise.
    Real JPEG is non-differentiable (quantization). We approximate it by:
    - Light blur (simulates DCT quantization smoothing)
    - Light noise (simulates quantization artifacts)
    Lower quality = more blur + more noise.
    """
    if quality is None:
        quality = random.randint(30, 80)
    # Map quality to blur sigma and noise sigma
    # quality=30: heavy (sigma=2.0, noise=0.03)
    # quality=80: light (sigma=0.3, noise=0.005)
    blur_sigma = 0.3 + (80 - quality) * 0.025  # 0.3 to 1.55
    noise_sigma = 0.005 + (80 - quality) * 0.0005  # 0.005 to 0.03

    images = diff_gaussian_blur(images, sigma=blur_sigma)
    noise = torch.randn_like(images) * noise_sigma
    return (images + noise).clamp(0, 1)


# Family grouping used only for the transform-family ablation (does not
# affect the default operator set below).
_TRANSFORM_FAMILY = {
    "blur": "degradation",
    "resize": "geometric",
    "noise": "degradation",
    "brightness": "photometric",
    "contrast": "photometric",
    "hflip": "geometric",
    "rotation": "geometric",
    "jpeg": "degradation",
}


# Base per-operator application probabilities (Table 1 of the paper).
_BASE_PROB = {
    "blur": 0.5, "resize": 0.5, "noise": 0.4, "brightness": 0.3,
    "contrast": 0.3, "hflip": 0.3, "rotation": 0.3, "jpeg": 0.4,
}


def sample_random_transform(families=None, prob_scale=1.0):
    """
    Sample a random composed transform.
    Each individual transform is applied with probability p.
    Returns a function that applies the composed transform.

    families: optional subset of {"degradation", "geometric", "photometric"}
        restricting which operators may be sampled (used only for the
        transform-family ablation). None (default) samples from the full
        operator set used for all main results.
    prob_scale: multiplies every base application probability, clamped to
        [0, 1] (used only for the probability-sensitivity ablation). 1.0
        (default) uses the base probabilities of Table 1.
    """
    def allowed(name):
        return families is None or _TRANSFORM_FAMILY[name] in families

    def p(name):
        return min(1.0, max(0.0, _BASE_PROB[name] * prob_scale))

    transforms_to_apply = []

    # Each transform applied with some probability
    if allowed("blur") and random.random() < p("blur"):
        transforms_to_apply.append(lambda x: diff_gaussian_blur(x))
    if allowed("resize") and random.random() < p("resize"):
        transforms_to_apply.append(lambda x: diff_resize(x))
    if allowed("noise") and random.random() < p("noise"):
        transforms_to_apply.append(lambda x: diff_noise(x))
    if allowed("brightness") and random.random() < p("brightness"):
        transforms_to_apply.append(lambda x: diff_brightness(x))
    if allowed("contrast") and random.random() < p("contrast"):
        transforms_to_apply.append(lambda x: diff_contrast(x))
    if allowed("hflip") and random.random() < p("hflip"):
        transforms_to_apply.append(lambda x: diff_hflip(x))
    if allowed("rotation") and random.random() < p("rotation"):
        transforms_to_apply.append(lambda x: diff_rotation(x))
    if allowed("jpeg") and random.random() < p("jpeg"):
        transforms_to_apply.append(lambda x: diff_jpeg_approx(x))

    # If nothing was selected, add at least one transform (restricted to
    # the allowed families, so a family-restricted run never falls back
    # to an excluded operator).
    if len(transforms_to_apply) == 0:
        fallback = [fn for name, fn in [
            ("blur", lambda x: diff_gaussian_blur(x)),
            ("resize", lambda x: diff_resize(x)),
            ("jpeg", lambda x: diff_jpeg_approx(x)),
        ] if allowed(name)] or [lambda x: diff_gaussian_blur(x)]
        transforms_to_apply.append(random.choice(fallback))

    def composed(x):
        for t in transforms_to_apply:
            x = t(x)
        return x

    return composed


# ============================================================
# EoT-TAA Attack
# ============================================================

def eot_taa_attack(
    images,
    backbone,
    mu,
    feature_type="patch",
    eps_inf=8.0 / 255.0,
    step_size=0.0004,
    num_steps=50,
    K=4,
    device="cuda",
    transform_reference=None,   # None -> EOT_TRANSFORM_REFERENCE env var (default: fixed clean)
    transform_families=None,    # None -> full operator set (see sample_random_transform)
    transform_prob_scale=1.0,   # 1.0 -> base probabilities (see sample_random_transform)
):
    """
    TR-TAA attack (Algorithm 1). Returns 8-bit-quantised adversarial images.

    transform_reference:
        False -> apply t_k only to x_adv; the reference z_o is the fixed clean
                 feature, computed once (TR-TAA, used for all paper results).
        True  -> apply the same t_k to both x_adv and x_o (transform-both
                 variant, used only in the reference-construction ablation).
        None  -> resolved from env var EOT_TRANSFORM_REFERENCE
                 ("1" => True, anything else / unset => False).

    transform_families:
        Optional subset of {"degradation", "geometric", "photometric"}
        restricting which operators are sampled at each step (used only for
        the transform-family ablation). None (default) uses the full
        8-operator set.

    transform_prob_scale:
        Multiplies every base application probability in
        Table 1, clamped to [0, 1] per
        operator (used only for the probability-sensitivity ablation).
        1.0 (default) reproduces the original probabilities exactly.
    """
    if transform_reference is None:
        transform_reference = os.environ.get("EOT_TRANSFORM_REFERENCE", "0") == "1"

    feature_extractor = VFMFeatureExtractor(backbone, feature_type)
    images = images.to(device)
    x_adv = images.clone().detach()

    # fixed clean reference, computed once from the untransformed image
    z_o_fixed_flat = None
    if not transform_reference:
        with torch.no_grad():
            x_o_norm = normalize_for_model(images)
            z_o0 = feature_extractor(x_o_norm)
            if z_o0.dim() == 2:
                z_o0 = z_o0 - mu.unsqueeze(0)
            else:
                z_o0 = z_o0 - mu.unsqueeze(0).unsqueeze(0)
            z_o_fixed_flat = z_o0.reshape(z_o0.shape[0], -1)

    for step in range(num_steps):
        total_grad = torch.zeros_like(images)

        for k in range(K):
            t_k = sample_random_transform(families=transform_families,
                                           prob_scale=transform_prob_scale)

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

            # reference: transform-both vs fixed clean
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
                z_o_flat = z_o_fixed_flat   # fixed clean reference

            cos_sim = F.cosine_similarity(z_adv_flat, z_o_flat, dim=-1).mean()

            grad_k = torch.autograd.grad(cos_sim, x_adv_k)[0]
            total_grad += grad_k.detach()

        avg_grad = total_grad / K
        x_adv = x_adv.detach() - step_size * avg_grad.sign()
        perturbation = torch.clamp(x_adv - images, min=-eps_inf, max=eps_inf)
        x_adv = torch.clamp(images + perturbation, min=0.0, max=1.0)

    # quantize to uint8 (simulate PNG storage)
    x_adv = (x_adv.detach() * 255.0).round() / 255.0
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv
