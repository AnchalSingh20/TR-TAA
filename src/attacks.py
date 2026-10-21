"""
Implementation of Task-Agnostic Attacks (TAA) and Task-Specific Attacks (TSA).

TAA: Algorithm 1 of Pulfer et al. (CVPRW 2025) - attacks the feature representation directly.
TSA: Standard PGD attacks targeting the downstream task (classification or segmentation).

Both use PGD as the solver with L∞ constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models import normalize_for_model, VFMFeatureExtractor


# ============================================================
# Task-Agnostic Attack (TAA) - Algorithm 1
# ============================================================

def compute_mean_features(backbone, dataloader, device, feature_type="class_and_patch",
                          num_samples=5000):
    """
    Compute the empirical mean vector μ from training images.
    (Lines 2-3 of Algorithm 1)

    Args:
        backbone: VFM backbone model
        dataloader: DataLoader for training images
        device: Device
        feature_type: Which tokens to use
        num_samples: Max number of samples to use

    Returns:
        mu: Mean feature vector
    """
    print(f"Computing mean features from {num_samples} samples...")
    feature_extractor = VFMFeatureExtractor(backbone, feature_type)

    all_features = []
    count = 0

    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc="Computing mean"):
            if count >= num_samples:
                break

            images = images.to(device)
            images_norm = normalize_for_model(images)

            features = feature_extractor(images_norm)

            # For patch/combined tokens, average over spatial dimension
            if features.dim() == 3:
                features = features.mean(dim=1)  # (B, D)

            all_features.append(features.cpu())
            count += images.shape[0]

    all_features = torch.cat(all_features, dim=0)[:num_samples]
    mu = all_features.mean(dim=0).to(device)

    print(f"Mean vector computed. Shape: {mu.shape}, Norm: {mu.norm():.4f}")
    return mu


def task_agnostic_attack(
    images,
    backbone,
    mu,
    feature_type="patch",
    eps_inf=8.0/255.0,
    step_size=0.0004,
    num_steps=50,
    device="cuda",
):
    """
    Task-Agnostic Attack (Algorithm 1 of Pulfer et al., CVPRW 2025).

    Minimizes cosine similarity between adversarial and original features
    in the mean-centered feature space.

    Args:
        images: Original images (B, 3, H, W) in [0, 1]
        backbone: Frozen VFM backbone
        mu: Mean feature vector for centering
        feature_type: 'class', 'patch', or 'class_and_patch'
        eps_inf: L∞ perturbation budget (ε∞)
        step_size: PGD step size (α)
        num_steps: Number of PGD steps (N)
        device: Device

    Returns:
        x_adv: Adversarial images (B, 3, H, W) in [0, 1], clipped & quantized
    """
    feature_extractor = VFMFeatureExtractor(backbone, feature_type)

    images = images.to(device)

    # Line 4: Initialize adversarial image
    x_adv = images.clone().detach()

    # Line 5: Compute target features (with stop gradient)
    with torch.no_grad():
        images_norm = normalize_for_model(images)
        z_o = feature_extractor(images_norm)  # (B, D) or (B, N, D)

        # Mean-center the target features
        if z_o.dim() == 2:
            z_o_tilde = z_o - mu.unsqueeze(0)
        else:  # (B, N, D)
            z_o_tilde = z_o - mu.unsqueeze(0).unsqueeze(0)

    # Lines 6-11: PGD loop
    for step in range(num_steps):
        x_adv.requires_grad_(True)

        # Line 7: Compute current features
        x_adv_norm = normalize_for_model(x_adv)
        z = feature_extractor(x_adv_norm)

        # Mean-center
        if z.dim() == 2:
            z_tilde = z - mu.unsqueeze(0)
        else:
            z_tilde = z - mu.unsqueeze(0).unsqueeze(0)

        # Line 8: Compute cosine similarity loss
        # For multi-token: flatten to (B, N*D) or average
        if z_tilde.dim() == 3:
            z_flat = z_tilde.reshape(z_tilde.shape[0], -1)
            z_o_flat = z_o_tilde.reshape(z_o_tilde.shape[0], -1)
        else:
            z_flat = z_tilde
            z_o_flat = z_o_tilde

        # Cosine similarity: we MINIMIZE this (push representations apart)
        cos_sim = F.cosine_similarity(z_flat, z_o_flat, dim=-1).mean()

        # Line 9: Gradient step (minimize cos_sim = subtract gradient)
        grad = torch.autograd.grad(cos_sim, x_adv)[0]
        x_adv = x_adv.detach() - step_size * grad.sign()

        # Line 10: Project back into ε∞ ball and [0, 1] range
        perturbation = torch.clamp(x_adv - images, min=-eps_inf, max=eps_inf)
        x_adv = torch.clamp(images + perturbation, min=0.0, max=1.0)

    # Post-processing: Quantize to uint8 and back (simulates saving as PNG)
    # TAA paper: "adversarial tensors are then clipped and quantized to get
    # real adversarial images, which are stored in png format and re-loaded"
    x_adv = (x_adv.detach() * 255.0).round() / 255.0
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


# ============================================================
# Task-Specific Attacks (TSA)
# ============================================================

def task_specific_attack_classification(
    images,
    labels,
    full_model,
    eps_inf=8.0/255.0,
    step_size=0.0004,
    num_steps=50,
    device="cuda",
):
    """
    Task-Specific Attack for classification (untargeted PGD).

    Maximizes cross-entropy loss to flip the predicted class.

    Args:
        images: Original images (B, 3, H, W) in [0, 1]
        labels: Ground truth labels (B,)
        full_model: Complete model (backbone + classification head)
        eps_inf, step_size, num_steps: PGD hyperparameters
        device: Device

    Returns:
        x_adv: Adversarial images
    """
    images = images.to(device)
    labels = labels.to(device)

    x_adv = images.clone().detach()
    criterion = nn.CrossEntropyLoss()

    for step in range(num_steps):
        x_adv.requires_grad_(True)

        logits = full_model(x_adv)
        loss = criterion(logits, labels)

        # Maximize loss (untargeted attack) = add gradient
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()

        # Project
        perturbation = torch.clamp(x_adv - images, min=-eps_inf, max=eps_inf)
        x_adv = torch.clamp(images + perturbation, min=0.0, max=1.0)

    # Quantize
    x_adv = (x_adv.detach() * 255.0).round() / 255.0
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def task_specific_attack_segmentation(
    images,
    masks,
    full_model,
    eps_inf=8.0/255.0,
    step_size=0.0004,
    num_steps=50,
    device="cuda",
):
    """
    Task-Specific Attack for segmentation (untargeted PGD).

    Maximizes pixel-wise cross-entropy loss.

    Args:
        images: Original images (B, 3, H, W) in [0, 1]
        masks: Ground truth segmentation masks (B, H, W)
        full_model: Complete model (backbone + segmentation head)
        eps_inf, step_size, num_steps: PGD hyperparameters
        device: Device

    Returns:
        x_adv: Adversarial images
    """
    images = images.to(device)
    masks = masks.to(device)

    x_adv = images.clone().detach()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for step in range(num_steps):
        x_adv.requires_grad_(True)

        logits = full_model(x_adv)  # (B, C, H, W)
        loss = criterion(logits, masks)

        # Maximize loss (untargeted)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()

        # Project
        perturbation = torch.clamp(x_adv - images, min=-eps_inf, max=eps_inf)
        x_adv = torch.clamp(images + perturbation, min=0.0, max=1.0)

    # Quantize
    x_adv = (x_adv.detach() * 255.0).round() / 255.0
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


# ============================================================
# PSNR computation
# ============================================================

def compute_psnr(original, adversarial):
    """
    Compute Peak Signal-to-Noise Ratio.

    From the TAA paper (Pulfer et al., 2025):
    PSNR = 10 * log10(255^2 / MSE)
    MSE = (1 / (3*H*W)) * ||x_o - x_a||^2

    where x ∈ [0, 1]^{H×W×3}

    Note: The TAA paper uses 255^2 in the numerator but images in [0,1],
    so effectively: PSNR = 10 * log10(1 / MSE) + 20*log10(255)
    Actually, looking more carefully: MSE is computed in [0,255] space.
    """
    # Convert to [0, 255] space
    orig_255 = original * 255.0
    adv_255 = adversarial * 255.0

    mse = ((orig_255 - adv_255) ** 2).mean()
    if mse == 0:
        return float('inf')

    psnr = 10 * torch.log10(torch.tensor(255.0 ** 2) / mse)
    return psnr.item()
