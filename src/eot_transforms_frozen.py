"""
Transformation sampler with frozen parameters (used by the VQA attack).

Samples the same eight-operator distribution as src/eot_attack.py (same
operators, application probabilities, parameter ranges and fallback rule),
but draws every parameter -- blur sigma, resize scale, noise level and noise
realisation, brightness/contrast factor, rotation angle, JPEG quality -- once,
when sample_random_transform() is called. The returned closure therefore
applies an identical transformation every time it is invoked.

It additionally supports an optional deployment resample appended to every
sampled transformation (deployment_resize), which models the image
processor of the victim model.

Self-test:
    python -m src.eot_transforms_frozen
"""

import random

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Primitive differentiable transforms
# Same maths as src/eot_attack.py; parameters are required arguments.
# ============================================================

def diff_gaussian_blur(images, sigma):
    """Differentiable Gaussian blur (depthwise conv2d, reflect padding)."""
    kernel_size = int(2 * round(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 21))

    k = torch.arange(kernel_size, device=images.device, dtype=torch.float32)
    k = k - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (k / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    gauss_2d = gauss_2d.to(images.dtype)
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    pad = kernel_size // 2
    padded = F.pad(images, [pad] * 4, mode="reflect")
    return F.conv2d(padded, gauss_2d, groups=3)


def diff_resize(images, scale):
    """Differentiable down-then-up resample (simulates resolution loss)."""
    B, C, H, W = images.shape
    small_h, small_w = max(4, int(H * scale)), max(4, int(W * scale))
    small = F.interpolate(images, size=(small_h, small_w), mode="bilinear",
                          align_corners=False)
    return F.interpolate(small, size=(H, W), mode="bilinear",
                         align_corners=False)


def diff_brightness(images, factor):
    return (images * factor).clamp(0, 1)


def diff_contrast(images, factor):
    mean = images.mean(dim=(-2, -1), keepdim=True)
    return ((images - mean) * factor + mean).clamp(0, 1)


def diff_hflip(images):
    return images.flip(-1)


def diff_rotation(images, angle_deg):
    """Differentiable rotation via affine grid + bilinear grid_sample."""
    angle_rad = angle_deg * np.pi / 180.0
    B = images.shape[0]
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))

    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        dtype=images.dtype, device=images.device,
    ).unsqueeze(0).expand(B, -1, -1)

    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def jpeg_params(quality):
    """Map a JPEG quality to the (blur_sigma, noise_sigma) approximation."""
    blur_sigma = 0.3 + (80 - quality) * 0.025
    noise_sigma = 0.005 + (80 - quality) * 0.0005
    return blur_sigma, noise_sigma


# ============================================================
# Frozen additive noise
# ============================================================

class _FrozenNoise:
    """
    Draws one noise tensor on first use and replays it thereafter, so
    repeated calls of the same sampled transform add the same noise.
    Keyed by (shape, device, dtype).
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self._cache = {}

    def __call__(self, images):
        key = (tuple(images.shape), images.device, images.dtype)
        if key not in self._cache:
            self._cache[key] = torch.randn(
                images.shape, device=images.device, dtype=images.dtype
            ) * self.sigma
        return (images + self._cache[key]).clamp(0, 1)


# ============================================================
# The sampler
# ============================================================

# Application probabilities (Table 1 of the paper), identical to
# src/eot_attack.py.
P_BLUR       = 0.5
P_RESIZE     = 0.5
P_NOISE      = 0.4
P_BRIGHTNESS = 0.3
P_CONTRAST   = 0.3
P_HFLIP      = 0.3
P_ROTATION   = 0.3
P_JPEG       = 0.4


def sample_random_transform(rng=None, deployment_resize=None):
    """
    Sample one composed transform with all parameters frozen.

    Args:
        rng: optional random.Random instance for reproducible sampling.
             The additive-noise tensor comes from torch's RNG.
        deployment_resize: optional int. If given, every composition ends
             with a resample through that resolution and back, modelling
             the victim model's image processor.

    Returns:
        composed(x) -> transformed tensor. Calling it twice applies the
        identical transform.
    """
    r = rng if rng is not None else random
    ops = []

    if r.random() < P_BLUR:
        sigma = r.uniform(0.5, 3.0)
        ops.append(lambda x, s=sigma: diff_gaussian_blur(x, sigma=s))

    if r.random() < P_RESIZE:
        scale = r.uniform(0.25, 0.75)
        ops.append(lambda x, sc=scale: diff_resize(x, scale=sc))

    if r.random() < P_NOISE:
        ops.append(_FrozenNoise(r.uniform(0.01, 0.05)))

    if r.random() < P_BRIGHTNESS:
        factor = r.uniform(0.7, 1.3)
        ops.append(lambda x, f=factor: diff_brightness(x, factor=f))

    if r.random() < P_CONTRAST:
        factor = r.uniform(0.7, 1.3)
        ops.append(lambda x, f=factor: diff_contrast(x, factor=f))

    if r.random() < P_HFLIP:
        ops.append(diff_hflip)

    if r.random() < P_ROTATION:
        angle = r.uniform(-30, 30)
        ops.append(lambda x, a=angle: diff_rotation(x, angle_deg=a))

    if r.random() < P_JPEG:
        quality = r.randint(30, 80)
        blur_sigma, noise_sigma = jpeg_params(quality)
        frozen = _FrozenNoise(noise_sigma)
        ops.append(
            lambda x, s=blur_sigma, fn=frozen: fn(diff_gaussian_blur(x, sigma=s))
        )

    if not ops:
        choice = r.choice(["blur", "resize", "jpeg"])
        if choice == "blur":
            s = r.uniform(0.5, 3.0)
            ops.append(lambda x, s=s: diff_gaussian_blur(x, sigma=s))
        elif choice == "resize":
            sc = r.uniform(0.25, 0.75)
            ops.append(lambda x, sc=sc: diff_resize(x, scale=sc))
        else:
            q = r.randint(30, 80)
            bs, ns = jpeg_params(q)
            fn = _FrozenNoise(ns)
            ops.append(lambda x, s=bs, fn=fn: fn(diff_gaussian_blur(x, sigma=s)))

    # Deployment resample of the victim's image processor.
    if deployment_resize is not None:
        def _deploy(x, target=int(deployment_resize)):
            H, W = x.shape[-2], x.shape[-1]
            up = F.interpolate(x, size=(target, target), mode="bicubic",
                               align_corners=False, antialias=False)
            return F.interpolate(up, size=(H, W), mode="bicubic",
                                 align_corners=False, antialias=True).clamp(0, 1)
        ops.append(_deploy)

    def composed(x):
        for op in ops:
            x = op(x)
        return x

    return composed


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(2, 3, 128, 128, device=dev)

    worst = 0.0
    for _ in range(200):
        t = sample_random_transform()
        d = (t(x) - t(x)).abs().max().item()
        worst = max(worst, d)
    print(f"max |t(x) - t(x)| over 200 sampled transforms: {worst:.3e} (expected 0)")

    t = sample_random_transform(deployment_resize=448)
    print("transform with deployment resample, output shape:", tuple(t(x).shape))
