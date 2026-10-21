"""
Vision Foundation Model (VFM) loading utilities.

Loads DINOv2 models via torch.hub (facebookresearch/dinov2).
Each model uses its default (ImageNet) normalization.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from functools import partial


# ============================================================
# ImageNet normalization (used by DiNOv2)
# ============================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dinov2_model(model_name: str, device: str = "cuda"):
    """
    Load a DiNOv2 model from torch.hub.

    Args:
        model_name: One of 'dinov2_vits14', 'dinov2_vitb14',
                    'dinov2_vitl14', 'dinov2_vitg14'
        device: Device to load model on

    Returns:
        model: The DiNOv2 backbone (frozen, eval mode)
    """
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device)
    model.eval()

    # Freeze all parameters - we never train the backbone
    for param in model.parameters():
        param.requires_grad = False

    return model


def get_model_info(model_name: str) -> dict:
    """Return model metadata matching Table 1."""
    info = {
        "dinov2_vits14": {"arch": "ViT-S/14", "params_m": 21, "dim": 384, "patch_size": 14},
        "dinov2_vitb14": {"arch": "ViT-B/14", "params_m": 85, "dim": 768, "patch_size": 14},
        "dinov2_vitl14": {"arch": "ViT-L/14", "params_m": 303, "dim": 1024, "patch_size": 14},
        "dinov2_vitg14": {"arch": "ViT-G/14", "params_m": 1136, "dim": 1536, "patch_size": 14},
    }
    return info[model_name]


def get_preprocessing_transform(image_size: int = 518):
    """
    Standard preprocessing for DiNOv2.
    DiNOv2 uses 518x518 input by default (37 patches of 14px each = 518).
    For PascalVOC, we resize to this size.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Note: We do NOT apply ImageNet normalization here.
        # Normalization is handled inside the attack/evaluation pipeline
        # so we can work with pixel-space perturbations in [0,1].
    ])


def normalize_for_model(images: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Apply ImageNet normalization to a batch of images.

    Args:
        images: Tensor of shape (B, 3, H, W) in [0, 1] range
        mean: Per-channel mean
        std: Per-channel std

    Returns:
        Normalized tensor
    """
    mean = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def denormalize_from_model(images: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization."""
    mean = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return images * std + mean


class VFMFeatureExtractor(nn.Module):
    """
    Wrapper around DiNOv2 that extracts features from different token types.

    For a ViT, the output contains:
    - CLS token: A single global representation vector
    - Patch tokens: Per-patch representations

    The TAA paper tests attacks on:
    1. Class token only
    2. Patch tokens only
    3. Class + patch tokens combined
    """

    def __init__(self, model, feature_type="class_and_patch"):
        """
        Args:
            model: DiNOv2 backbone
            feature_type: One of 'class', 'patch', 'class_and_patch'
        """
        super().__init__()
        self.model = model
        self.feature_type = feature_type

    def forward(self, x):
        """
        Extract features based on the specified token type.

        Args:
            x: Normalized input tensor (B, 3, H, W)

        Returns:
            features: Tensor whose shape depends on feature_type:
                - 'class': (B, D) - CLS token only
                - 'patch': (B, N_patches, D) - all patch tokens
                - 'class_and_patch': (B, 1 + N_patches, D) - CLS + patches
        """
        # DiNOv2 forward_features returns a dict with different outputs
        output = self.model.forward_features(x)

        cls_token = output["x_norm_clstoken"]      # (B, D)
        patch_tokens = output["x_norm_patchtokens"]  # (B, N_patches, D)

        if self.feature_type == "class":
            return cls_token
        elif self.feature_type == "patch":
            return patch_tokens
        elif self.feature_type == "class_and_patch":
            # Concatenate CLS token with patch tokens along sequence dim
            cls_expanded = cls_token.unsqueeze(1)  # (B, 1, D)
            return torch.cat([cls_expanded, patch_tokens], dim=1)  # (B, 1+N, D)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
