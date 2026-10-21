"""
Scale-Invariant Log Loss (SiLogLoss) for monocular depth.

Reference: Eigen et al., "Depth Map Prediction from a Single Image using
a Multi-Scale Deep Network", NeurIPS 2014. Section 3.2 ("Scale-Invariant
Error").

The intuition:
  - Monocular depth is ambiguous up to a global scale (a small object up
    close and a large object far away can produce the same image).
  - SiLogLoss measures the depth error AFTER removing the optimal global
    log-shift; equivalently, it penalizes the variance of the log-error
    across pixels but not the global mean.
  - Modern monocular depth networks (DPT, AdaBins, NeWCRFs, Depth Anything)
    train with SiLogLoss as the primary loss.

The standard form is:

    L = sqrt( var(d_i) + lambda * mean(d_i)^2 )

where d_i = log(pred_i) - log(gt_i) for each valid pixel i.

  - lambda controls how much the global scale matters. lambda=0 is
    fully scale-invariant; lambda=1 is plain log-RMSE. The common
    choice in the literature is lambda = 0.85, which we use as default.
  - We multiply by an overall factor of 10 (matching BTS / DPT
    conventions) so the loss is in a similar numerical range to other
    components when combined with other losses.

We compute the loss ONLY on valid pixels (gt > 0) and skip batches that
end up with zero valid pixels (extremely rare but defended against).
"""

import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    """
    Scale-Invariant Log Loss.

    Args:
        lambda_var: weight of the variance term. 0.85 is the BTS/DPT
                    convention. Set 1.0 for plain log-RMSE.
        scale: overall scale factor (default 10.0, matches BTS).
        min_depth: depth values below this are treated as invalid even
                   if valid_mask says otherwise. Prevents log(0) and
                   numerical issues. 0.001 m matches NYUv2 clipping.
    """

    def __init__(self, lambda_var=0.85, scale=10.0, min_depth=0.001):
        super().__init__()
        self.lambda_var = lambda_var
        self.scale = scale
        self.min_depth = min_depth

    def forward(self, log_pred, gt_meters, valid_mask):
        """
        Args:
            log_pred: (B, H, W) predicted LOG-DEPTH (from LinearDepthHead).
                      Same shape as gt_meters.
            gt_meters: (B, H, W) ground-truth depth in meters (0 = invalid)
            valid_mask: (B, H, W) bool tensor, True where gt is valid
        Returns:
            scalar loss
        """
        if log_pred.shape != gt_meters.shape:
            raise ValueError(
                f"log_pred {log_pred.shape} and gt_meters {gt_meters.shape} "
                f"must have the same shape."
            )

        # Final mask: provided valid_mask AND gt >= min_depth
        mask = valid_mask & (gt_meters >= self.min_depth)

        if mask.sum() == 0:
            # Defensive: no valid pixels means no learning signal for
            # this batch. Return a zero loss that has requires_grad
            # connected to the prediction so backward doesn't crash.
            return (log_pred * 0).sum()

        # Per-pixel log differences (in log-meters)
        gt_log = torch.log(gt_meters.clamp(min=self.min_depth))
        d = log_pred[mask] - gt_log[mask]

        # SiLog: sqrt( mean(d^2) - lambda * mean(d)^2 )
        # When lambda=1, this is plain log-RMSE (zero-mean adjusted).
        # When lambda=0.85, the global shift can vary by ~15% without
        # penalty, which matches monocular depth's natural ambiguity.
        d_var = (d * d).mean()
        d_mean = d.mean()
        loss = torch.sqrt(d_var - self.lambda_var * d_mean * d_mean
                          + 1e-12)        # +epsilon for sqrt safety

        return self.scale * loss
