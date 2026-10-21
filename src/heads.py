"""
Downstream task heads for classification and segmentation.

These are the heads trained on top of the frozen VFM backbone.

Classification: Linear head on the CLS token (standard linear probe).

Segmentation: following Section 6.1 of the TAA paper (Pulfer et al., 2025):
    "We concatenate the CLS token to every patch token along the feature
    dimension and use bi-cubic interpolation followed by a 2-dimensional
    convolution to obtain predictions for the original image size."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassificationHead(nn.Module):
    """
    Simple linear classification head on top of VFM CLS token.

    This is the standard linear probing setup used in the TAA paper:
    - Input: CLS token features (B, D)
    - Output: Class logits (B, num_classes)
    """

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        """
        Args:
            features: CLS token (B, D)
        Returns:
            logits: (B, num_classes)
        """
        return self.linear(features)


class SegmentationHead(nn.Module):
    """
    Segmentation head of the TAA paper (Section 6.1):
    1. Concatenate CLS token to every patch token along feature dimension
    2. Apply bi-cubic interpolation to upsample to original image size
    3. Apply 2D convolution to get per-pixel class predictions

    Architecture details:
    - Input patch tokens: (B, N_patches, D) where N_patches = (H/patch_size)^2
    - CLS token: (B, D)
    - After concatenation: (B, N_patches, 2*D)
    - Reshape to spatial: (B, 2*D, H/patch_size, W/patch_size)
    - Upsample to: (B, 2*D, H, W)
    - Conv2d to: (B, num_classes, H, W)
    """

    def __init__(self, feature_dim, num_classes, patch_size=14, image_size=518):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches_per_side = image_size // patch_size

        # After concatenating CLS to each patch: input dim = 2 * feature_dim
        # Use 1x1 conv to reduce to num_classes
        self.conv = nn.Conv2d(2 * feature_dim, num_classes, kernel_size=1)

    def forward(self, cls_token, patch_tokens):
        """
        Args:
            cls_token: (B, D)
            patch_tokens: (B, N_patches, D)

        Returns:
            logits: (B, num_classes, image_size, image_size)
        """
        B, N, D = patch_tokens.shape
        H_p = W_p = self.num_patches_per_side

        # Expand CLS token and concatenate to each patch token
        # CLS: (B, D) -> (B, N, D)
        cls_expanded = cls_token.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: (B, N, 2*D)
        combined = torch.cat([cls_expanded, patch_tokens], dim=-1)

        # Reshape to spatial grid: (B, 2*D, H_p, W_p)
        combined = combined.permute(0, 2, 1).reshape(B, 2 * D, H_p, W_p)

        # Bi-cubic interpolation to full image size
        combined = F.interpolate(
            combined,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )

        # 1x1 convolution to get per-pixel predictions
        logits = self.conv(combined)  # (B, num_classes, H, W)

        return logits


class FullModel(nn.Module):
    """
    Complete model: VFM backbone + downstream head.
    Used for Task-Specific Attacks which need gradients through the full pipeline.
    """

    def __init__(self, backbone, head, task="classification", normalize_fn=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.task = task
        self.normalize_fn = normalize_fn

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W) in [0, 1] range

        Returns:
            Task-specific output
        """
        # Normalize input
        if self.normalize_fn is not None:
            x_norm = self.normalize_fn(x)
        else:
            x_norm = x

        # Get features from backbone
        features = self.backbone.forward_features(x_norm)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]

        if self.task == "classification":
            return self.head(cls_token)
        elif self.task == "segmentation":
            return self.head(cls_token, patch_tokens)
        else:
            raise ValueError(f"Unknown task: {self.task}")
