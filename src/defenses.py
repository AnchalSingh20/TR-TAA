"""
src/defenses.py

Test-time input-preprocessing defenses for the TR-TAA experiments.

Two defenses:
  1. jpeg_defense(imgs, quality)  -- aggressive JPEG recompression
  2. DiffPure (class)             -- diffusion-based purification
                                    (Nie et al., 2022), diffusers backend

Both operate on image tensors in [0,1] with shape (B, 3, H, W). The cached
PascalVOC images are 518x518 (DINOv2 native). The diffusion purifier runs
at its native 256x256, so DiffPure downsamples 518->256, purifies, then
upsamples 256->518 before returning. This up/downsample is itself a mild
transformation that affects clean and adversarial images alike; the
quantity of interest is the RELATIVE attack strength retained through
purification (does TR-TAA survive better than TAA),
not the absolute post-defense number.

DiffPure mechanism:
  - choose a purification strength t* in (0, 1], mapped to a diffusion
    timestep T* = round(t* * (num_train_timesteps - 1))
  - add forward diffusion noise to bring x to the marginal q(x_{T*} | x)
  - run the reverse denoiser from T* back to 0
Smaller t* = lighter purification (preserves more of the image, removes
less perturbation); larger t* = heavier purification. The DiffPure paper
uses t* around 0.1-0.15 for L_inf attacks; we default to 0.1.
"""

import io
import random
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


# ============================================================
# JPEG defense
# ============================================================
def jpeg_defense(imgs, quality=75):
    """
    Re-encode each image as JPEG at the given quality and decode back.
    imgs: (B,3,H,W) float in [0,1]. Returns same shape/device.
    """
    out = []
    for i in range(imgs.shape[0]):
        pil = TF.to_pil_image(imgs[i].detach().cpu().clamp(0, 1))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        dec = Image.open(buf).convert("RGB")
        out.append(TF.to_tensor(dec))
    return torch.stack(out).to(imgs.device)


# ============================================================
# Bit-depth reduction (feature squeezing; Xu, Evans & Qi 2018)
# ============================================================
def bit_depth_reduction(imgs, bits=4):
    """
    Quantize each channel to `bits` bits per pixel.
    imgs: (B,3,H,W) float in [0,1]. Returns same shape/device.
    """
    levels = (2 ** int(bits)) - 1
    return torch.round(imgs.clamp(0, 1) * levels) / levels


# ============================================================
# Randomized resizing + padding (Xie et al. 2018, ICLR)
# ============================================================
def randomized_resize_pad(imgs, min_scale=0.9, max_scale=1.0):
    """
    Resize to a random scale in [min_scale, max_scale] of the original
    (H, W), then pad back to the original (H, W) at a random offset.
    imgs: (B,3,H,W) float in [0,1]. Returns same shape/device.
    """
    B, C, H, W = imgs.shape
    scale = random.uniform(min_scale, max_scale)
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    resized = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear",
                            align_corners=False)

    pad_h, pad_w = H - new_h, W - new_w
    pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
    pad_left = random.randint(0, pad_w) if pad_w > 0 else 0
    pad_bottom = pad_h - pad_top
    pad_right = pad_w - pad_left
    padded = F.pad(resized, [pad_left, pad_right, pad_top, pad_bottom],
                   mode="constant", value=0.0)
    return padded.clamp(0, 1)


# ============================================================
# Median filter / spatial smoothing (Xu et al. 2018; Guo et al. 2018)
# ============================================================
def median_filter_defense(imgs, size=3):
    """
    Apply a size x size median filter to each image (denoising-based
    purification, a cheap stand-in for full TV minimization).
    imgs: (B,3,H,W) float in [0,1]. Returns same shape/device.
    """
    out = []
    for i in range(imgs.shape[0]):
        pil = TF.to_pil_image(imgs[i].detach().cpu().clamp(0, 1))
        filtered = pil.filter(ImageFilter.MedianFilter(size=int(size)))
        out.append(TF.to_tensor(filtered))
    return torch.stack(out).to(imgs.device)


# ============================================================
# Multi-level Floyd-Steinberg dithering (error-diffusion quantization)
# ============================================================
def _uniform_grid_palette(levels=4):
    """
    Build a PIL 'P'-mode palette image whose colors are an evenly spaced
    `levels`-per-channel RGB grid (levels^3 colors, must be <= 256).
    """
    assert levels ** 3 <= 256, "levels^3 must fit in an 8-bit palette"
    vals = [round(i * 255 / (levels - 1)) for i in range(levels)]
    colors = [(r, g, b) for r in vals for g in vals for b in vals]
    flat = [c for rgb in colors for c in rgb]
    flat += [0] * (768 - len(flat))  # pad to 256 * 3 palette entries
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat)
    return pal_img


def dither_defense(imgs, levels=4):
    """
    Multi-level Floyd-Steinberg dithering (Floyd & Steinberg, 1976):
    quantize each pixel to the nearest color on an evenly spaced
    `levels`-per-channel grid, diffusing the quantization error to
    unprocessed neighbors instead of simply rounding (bit_depth_reduction).
    The error diffusion preserves visual structure at aggressive
    quantization, which is the point of testing it as a defense.
    imgs: (B,3,H,W) float in [0,1]. Returns same shape/device.
    """
    palette = _uniform_grid_palette(levels)
    out = []
    for i in range(imgs.shape[0]):
        pil = TF.to_pil_image(imgs[i].detach().cpu().clamp(0, 1))
        dithered = pil.quantize(
            palette=palette, dither=Image.Dither.FLOYDSTEINBERG
        ).convert("RGB")
        out.append(TF.to_tensor(dithered))
    return torch.stack(out).to(imgs.device)


# ============================================================
# DiffPure (diffusers backend)
# ============================================================
class DiffPure:
    """
    Diffusion-based input purification.

    Loads an unconditional pixel-space DDPM from the HuggingFace hub via
    diffusers. Tries an ImageNet-256 checkpoint first (closest domain to
    PascalVOC), then falls back to an LSUN scene model, then CelebA-HQ.
    The actually-loaded model id is stored in self.loaded_model_id so the
    caller can report/cite it.

    Usage:
        dp = DiffPure(device="cuda", t_star=0.1)
        purified = dp(imgs)          # imgs (B,3,H,W) in [0,1] at any HxW
    """

    # Candidate unconditional 256x256 pixel-space DDPMs, in order of
    # domain preference for natural multi-object images.
    CANDIDATES = [
        "google/ddpm-imagenet-256",     # best domain match (may be absent)
        "google/ddpm-church-256",       # LSUN scenes (natural images)
        "google/ddpm-bedroom-256",      # LSUN scenes
        "google/ddpm-celebahq-256",     # faces (last resort)
    ]

    def __init__(self, device="cuda", t_star=0.1, diffusion_size=256,
                 num_inference_steps=None, model_id=None, seed=0):
        from diffusers import DDPMScheduler, UNet2DModel

        self.device = device
        self.t_star = float(t_star)
        self.diffusion_size = diffusion_size
        self.seed = seed

        ids_to_try = [model_id] if model_id else self.CANDIDATES
        last_err = None
        self.unet = None
        for mid in ids_to_try:
            try:
                print(f"[DiffPure] trying checkpoint: {mid} ...")
                self.unet = UNet2DModel.from_pretrained(mid).to(device).eval()
                self.scheduler = DDPMScheduler.from_pretrained(mid)
                self.loaded_model_id = mid
                print(f"[DiffPure] loaded: {mid}")
                break
            except Exception as e:  # noqa
                last_err = e
                print(f"[DiffPure]   unavailable ({type(e).__name__}); trying next.")
        if self.unet is None:
            raise RuntimeError(
                f"Could not load any DiffPure checkpoint. Last error: {last_err}"
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_star_step = int(round(self.t_star * (self.num_train_timesteps - 1)))
        # Optionally subsample reverse steps for speed (DDPM is sequential).
        self.num_inference_steps = num_inference_steps
        print(f"[DiffPure] t*={self.t_star} -> timestep {self.t_star_step} "
              f"of {self.num_train_timesteps}")

    @torch.no_grad()
    def _purify_256(self, x256):
        """
        x256: (B,3,256,256) in [0,1]. Returns purified (B,3,256,256) in [0,1].
        Diffusion model expects inputs in [-1, 1].
        """
        device = self.device
        B = x256.shape[0]
        x = x256 * 2.0 - 1.0  # [0,1] -> [-1,1]

        # Forward diffusion to t*: q(x_t | x_0) = sqrt(acp) x0 + sqrt(1-acp) eps
        acp = self.scheduler.alphas_cumprod.to(device)[self.t_star_step]
        gen = torch.Generator(device=device).manual_seed(self.seed)
        noise = torch.randn(x.shape, generator=gen, device=device, dtype=x.dtype)
        x_t = (acp ** 0.5) * x + ((1.0 - acp) ** 0.5) * noise

        # Build the reverse timestep schedule from t_star_step down to 0.
        if self.num_inference_steps is not None:
            self.scheduler.set_timesteps(self.num_inference_steps, device=device)
            timesteps = [t for t in self.scheduler.timesteps if t <= self.t_star_step]
        else:
            timesteps = list(range(self.t_star_step, -1, -1))

        # Reverse denoising loop.
        xt = x_t
        for t in timesteps:
            t_tensor = torch.full((B,), int(t), device=device, dtype=torch.long)
            model_out = self.unet(xt, t_tensor).sample
            step = self.scheduler.step(model_out, int(t), xt)
            xt = step.prev_sample

        x0 = (xt.clamp(-1, 1) + 1.0) / 2.0  # back to [0,1]
        return x0.clamp(0, 1)

    @torch.no_grad()
    def __call__(self, imgs):
        """
        imgs: (B,3,H,W) in [0,1] at any resolution.
        Returns purified imgs at the SAME (H,W) as input.
        """
        H, W = imgs.shape[-2:]
        x = imgs.to(self.device)
        # downsample to diffusion resolution
        if (H, W) != (self.diffusion_size, self.diffusion_size):
            x256 = F.interpolate(x, size=(self.diffusion_size, self.diffusion_size),
                                 mode="bilinear", align_corners=False)
        else:
            x256 = x
        x256p = self._purify_256(x256.clamp(0, 1))
        # upsample back to original resolution
        if (H, W) != (self.diffusion_size, self.diffusion_size):
            xp = F.interpolate(x256p, size=(H, W), mode="bilinear",
                               align_corners=False)
        else:
            xp = x256p
        return xp.clamp(0, 1)


# ============================================================
# Defense registry helper
# ============================================================
def build_defenses(args, device):
    """
    Returns an ordered dict {name: callable(imgs)->imgs}.

    'No defense' is always present.
    JPEG defenses: pass args.jpeg_qualities (list of ints, e.g. [75] or [75, 50]).
                   For backward compatibility, args.jpeg_quality (single int)
                   is still accepted if jpeg_qualities is absent.
    Bit-depth reduction: args.bit_depth (default 4 bits); skip via
                   args.no_bitdepth.
    Randomized resize+pad: args.resize_pad_min_scale (default 0.9); skip via
                   args.no_resize_pad.
    Median filter: args.median_size (default 3); skip via args.no_median.
    Dithering: args.dither_levels (default 4); skip via args.no_dither.
    DiffPure is constructed lazily (loads a diffusion model) only if requested
    via NOT --no_diffpure.
    """
    from collections import OrderedDict
    defenses = OrderedDict()
    defenses["No defense"] = (lambda x: x)

    # ------- JPEG (one entry per requested quality) -------
    qualities = getattr(args, "jpeg_qualities", None)
    if qualities is None:
        qualities = [getattr(args, "jpeg_quality", 75)]
    for q in qualities:
        # IMPORTANT: bind q at lambda-creation time via the default-arg trick;
        # without `q=q`, every lambda captures the SAME (last) loop variable.
        defenses[f"JPEG q={q}"] = (lambda x, q=q: jpeg_defense(x, quality=q))

    # ------- Bit-depth reduction -------
    if not getattr(args, "no_bitdepth", False):
        bits = getattr(args, "bit_depth", 4)
        defenses[f"BitDepth b={bits}"] = (
            lambda x, bits=bits: bit_depth_reduction(x, bits=bits))

    # ------- Randomized resize + pad -------
    if not getattr(args, "no_resize_pad", False):
        min_scale = getattr(args, "resize_pad_min_scale", 0.9)
        defenses[f"ResizePad s=[{min_scale},1.0]"] = (
            lambda x, min_scale=min_scale: randomized_resize_pad(x, min_scale=min_scale))

    # ------- Median filter -------
    if not getattr(args, "no_median", False):
        msize = getattr(args, "median_size", 3)
        defenses[f"Median k={msize}"] = (
            lambda x, msize=msize: median_filter_defense(x, size=msize))

    # ------- Dithering (multi-level Floyd-Steinberg) -------
    if not getattr(args, "no_dither", False):
        dlevels = getattr(args, "dither_levels", 4)
        defenses[f"DitherFS L={dlevels}"] = (
            lambda x, dlevels=dlevels: dither_defense(x, levels=dlevels))

    # ------- DiffPure (optional; constructs a UNet) -------
    if not args.no_diffpure:
        dp = DiffPure(
            device=device,
            t_star=args.diffpure_tstar,
            num_inference_steps=args.diffpure_steps,
            model_id=(args.diffpure_model or None),
        )
        name = f"DiffPure t*={args.diffpure_tstar} ({dp.loaded_model_id.split('/')[-1]})"
        defenses[name] = dp
    return defenses
