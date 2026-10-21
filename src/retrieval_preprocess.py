"""
src/retrieval_preprocess.py
===========================
Single source of truth for R-Oxford preprocessing and descriptors.

Both the clean-baseline script (evaluate_clean_baseline_roxford.py), the dataset
(datasets_roxford.py), and the attack evaluators import from HERE, so clean,
attacked-generation, and attacked-evaluation all use byte-identical preprocessing.
That consistency is what makes the clean-vs-attack drop valid.

Functions: _round14, load_image_aspect, signed_gem, pool_descriptor,
extract_from_disk, fit_pcaw, apply_pcaw.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.models import normalize_for_model

# ImageNet mean colour used for letterbox padding (0-255).
# Padding contributes almost nothing to the descriptor, but keep one value.
_IMAGENET_PAD_255 = (124, 116, 104)


def _round14(x):
    x = int(round(x))
    return max(14, (x // 14) * 14)


def load_image_aspect(path, image_size, bbox=None, pad="square"):
    """
    Load an image (optionally bbox-cropped for queries) and resize with the
    LONGER side -> image_size, preserving aspect ratio, BICUBIC.

    pad == "square": paste onto an image_size x image_size canvas filled with
                     the ImageNet mean colour -> fixed (3, S, S), batchable.
    pad == "none":   no padding; both sides rounded to a multiple of 14 ->
                     variable (3, H, W), must run one image at a time.
    Returns a float tensor in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        w, h = img.size
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        img = img.crop((x1, y1, x2, y2))
    w, h = img.size
    scale = image_size / max(w, h)
    if pad == "square":
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (image_size, image_size), _IMAGENET_PAD_255)
        canvas.paste(img, ((image_size - new_w) // 2, (image_size - new_h) // 2))
        return TF.to_tensor(canvas)
    elif pad == "none":
        new_w = _round14(w * scale)
        new_h = _round14(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        return TF.to_tensor(img)
    else:
        raise ValueError(f"unknown pad mode {pad}")


# ============================================================
# Descriptors (signed GeM handles DINOv2's negative LayerNorm features)
# ============================================================
def signed_gem(patch, p=3.0, eps=1e-6):
    s = patch.sign()
    powered = s * patch.abs().clamp(min=eps).pow(p)
    m = powered.mean(dim=1)
    return m.sign() * m.abs().clamp(min=eps).pow(1.0 / p)


def pool_descriptor(out, descriptor, p):
    cls = out["x_norm_clstoken"]
    patch = out["x_norm_patchtokens"]
    if descriptor == "cls":
        feat = cls
    elif descriptor == "mean":
        feat = patch.mean(dim=1)
    elif descriptor == "gem":
        feat = signed_gem(patch, p=p)
    elif descriptor == "cls_patch":
        feat = torch.cat(
            [F.normalize(cls, p=2, dim=1),
             F.normalize(patch.mean(dim=1), p=2, dim=1)], dim=1)
    else:
        raise ValueError(descriptor)
    return F.normalize(feat, p=2, dim=1)


# ============================================================
# Feature extraction from a list of image paths (+ optional bboxes)
# ============================================================
def extract_from_disk(backbone, entries, image_size, scales, descriptor, p,
                      pad, device, batch_size, desc="extract"):
    """
    entries: list of (path, bbox_or_None)
    Returns (N, D) numpy array of RAW (pre-whitening) L2-normalized descriptors,
    multi-scale averaged.
    """
    N = len(entries)
    per_scale = []
    for s in scales:
        size = _round14(image_size * s)
        feats = []
        if pad == "none":
            for (path, bbox) in tqdm(entries, desc=f"{desc}|scale{s:.2f}|var",
                                     leave=False):
                img = load_image_aspect(path, size, bbox=bbox, pad="none")
                x = normalize_for_model(img.unsqueeze(0).to(device))
                with torch.no_grad():
                    out = backbone.forward_features(x)
                feats.append(pool_descriptor(out, descriptor, p).cpu())
            per_scale.append(torch.cat(feats, dim=0))
        else:
            imgs = torch.stack(
                [load_image_aspect(p_, size, bbox=b_, pad="square")
                 for (p_, b_) in entries], dim=0)
            for st in tqdm(range(0, N, batch_size),
                           desc=f"{desc}|scale{s:.2f}({size}px)", leave=False):
                ed = min(st + batch_size, N)
                x = normalize_for_model(imgs[st:ed].to(device))
                with torch.no_grad():
                    out = backbone.forward_features(x)
                feats.append(pool_descriptor(out, descriptor, p).cpu())
                del x
            per_scale.append(torch.cat(feats, dim=0))
    feat = torch.stack(per_scale, dim=0).mean(dim=0)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.numpy()


# ============================================================
# PCA whitening (fit on clean DB, apply to queries + DB)
# ============================================================
def fit_pcaw(X, dims=None, shrink=1e-5):
    X = X.astype(np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / Xc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    if dims is not None:
        eigvals, eigvecs = eigvals[:dims], eigvecs[:, :dims]
    P = eigvecs / np.sqrt(eigvals + shrink)
    return mean.astype(np.float32), P.astype(np.float32)


def apply_pcaw(X, mean, P):
    Y = (X - mean) @ P
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Y.astype(np.float32)
