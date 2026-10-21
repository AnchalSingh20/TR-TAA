"""
Linear depth head for DINOv2 ViT backbones.

Architecture (matches the DINOv2 paper's linear-probe depth recipe,
Oquab et al. 2024, Appendix):

    patch tokens (B, N, D)
       -> reshape to spatial grid (B, D, h_p, w_p)
       -> 1x1 Conv2d to 1 channel (B, 1, h_p, w_p)
       -> bilinear upsample to (B, 1, H_out, W_out)

The single learned layer is the 1x1 Conv2d, with D inputs and 1 output.
For ViT-B/14 with D=768, the head has 768 + 1 = 769 parameters.

We predict LOG-DEPTH (the network output is in log-meters space). At
evaluation time we exponentiate to get depth in meters. Training uses
the scale-invariant log loss (SiLogLoss) from Eigen et al. 2014, which
is the dominant choice in monocular depth literature.

Why log-depth + SiLog:
  - Depth has a long-tailed distribution (many near pixels, few far).
    Linear MSE is dominated by far-pixel errors; log-space equalizes.
  - SiLogLoss is invariant to a global scale shift, which monocular
    depth methods are fundamentally ambiguous to.
  - The DINOv2 paper uses exactly this setup for their linear-probe
    NYUv2 numbers, so our results are comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDepthHead(nn.Module):
    """
    1x1 Conv2d head on top of frozen ViT patch tokens.

    Output is in LOG-DEPTH space. Exponentiate (and clip to a valid depth
    range) to recover depth in meters.

    Args:
        feature_dim: D, the patch-token feature dimension (384 for ViT-S,
                     768 for ViT-B, 1024 for ViT-L, 1536 for ViT-G)
        patch_size:  patch dimension of the ViT (14 for DINOv2 /14 models)
        image_size:  input image side length (518 for our pipeline)
        output_size: target output spatial size (480, 640) for NYUv2

    Notes:
        - We accept the raw patch token tensor (B, N, D) from
          backbone.forward_features(x)['x_norm_patchtokens'] and reshape
          it into a spatial grid. N = (image_size / patch_size)^2.
        - The 1x1 conv maps D -> 1; bilinear upsample produces dense
          per-pixel log-depth at output_size.
    """

    def __init__(self, feature_dim, patch_size=14, image_size=518,
                 output_size=(480, 640)):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.output_size = output_size

        # Spatial grid side: 518/14 = 37 for our default
        self.h_p = image_size // patch_size
        self.w_p = image_size // patch_size
        self.n_tokens = self.h_p * self.w_p

        # The only learned layer
        self.proj = nn.Conv2d(feature_dim, 1, kernel_size=1)

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, N, D) where N must equal self.n_tokens
        Returns:
            (B, output_H, output_W) tensor of LOG-DEPTH predictions
            (apply torch.exp(...) and clip for depth in meters)
        """
        B, N, D = patch_tokens.shape
        if N != self.n_tokens:
            raise ValueError(
                f"Expected {self.n_tokens} patch tokens "
                f"({self.h_p}x{self.w_p}); got {N}. Check image_size "
                f"and patch_size are consistent with the backbone."
            )
        if D != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {D}."
            )

        # (B, N, D) -> (B, D, h_p, w_p)
        x = patch_tokens.transpose(1, 2).reshape(B, D, self.h_p, self.w_p)

        # 1x1 conv: D -> 1
        x = self.proj(x)                              # (B, 1, h_p, w_p)

        # Upsample to native depth resolution
        x = F.interpolate(
            x, size=self.output_size,
            mode="bilinear", align_corners=False
        )                                              # (B, 1, H, W)

        return x.squeeze(1)                            # (B, H, W) log-depth


def log_depth_to_meters(log_depth, min_depth=0.001, max_depth=10.0):
    """
    Convert log-depth predictions (from LinearDepthHead) to meters,
    clipped to the valid NYUv2 indoor range.

    Args:
        log_depth: torch tensor of any shape, predicted log-depth
        min_depth, max_depth: clip range in meters
    Returns:
        same-shape tensor of depth in meters
    """
    depth_m = torch.exp(log_depth)
    return depth_m.clamp(min=min_depth, max=max_depth)
