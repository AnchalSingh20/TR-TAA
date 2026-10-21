"""
Standard depth prediction metrics for NYU Depth v2.

Reference: Eigen et al., NeurIPS 2014. Every modern depth-prediction
paper uses this exact metric set.

All metrics expect:
  pred: (H, W) or (N, H, W) torch tensor of predicted depth in METERS
  gt:   (H, W) or (N, H, W) torch tensor of ground-truth depth in METERS
  valid_mask: (H, W) or (N, H, W) bool tensor, True where gt > 0

Pixels with gt == 0 (invalid sensor readings in NYUv2) MUST be excluded.
Including them would corrupt every metric.

Standard evaluation also CLIPS depth to a valid range, typically
[0.001, 10] meters for NYUv2, to avoid divide-by-near-zero in AbsRel
and to match what the depth sensor can actually measure.

Standard NYUv2 Eigen crop.
  The canonical Eigen protocol evaluates only on the center crop
  [45:471, 41:601] of the native 480x640 grid, discarding the noisy
  sensor border where depth is unreliable. This is the protocol used by
  the DINOv2 / DINOv3 linear-probe depth numbers, so it is required for
  a fair comparison. Applying it raises delta1 and lowers RMSE without
  changing the RELATIVE attack effect (it is applied identically to
  clean and adversarial images). use_eigen_crop=False evaluates the
  full frame instead.

Returned metric values:
  RMSE:    Root Mean Square Error in meters. Lower is better.
  RMSE_log: RMSE in log-depth space. Lower is better.
  AbsRel:  Mean |pred - gt| / gt. Lower is better.
  SqRel:   Mean (pred - gt)^2 / gt. Lower is better.
  delta1:  Fraction of pixels where max(pred/gt, gt/pred) < 1.25.
           Higher is better. (Standard "accuracy" threshold.)
  delta2:  Same with threshold 1.25^2 = 1.5625.
  delta3:  Same with threshold 1.25^3 = 1.953125.

For the attack experiments:
  - Stronger attack = HIGHER RMSE, AbsRel, SqRel (depth gets worse)
  - Stronger attack = LOWER delta1, delta2, delta3 (fewer accurate pixels)
"""

import torch
import torch.nn.functional as F


# Standard NYUv2 evaluation depth clipping range, in meters.
# Predictions and ground truth are both clipped to this range before
# metrics are computed.
NYUV2_MIN_DEPTH = 0.001
NYUV2_MAX_DEPTH = 10.0

# Standard NYUv2 Eigen center crop on the native 480x640 grid:
# (top, bottom, left, right). Rows [45:471], cols [41:601].
EIGEN_CROP = (45, 471, 41, 601)


def align_pred_to_gt(pred, gt):
    """
    Resize a (potentially low-res) prediction up to the ground-truth
    resolution using bilinear interpolation. The standard depth
    evaluation pipeline always evaluates at the GT resolution.

    Args:
        pred: (H', W') or (N, H', W') torch tensor (predicted depth)
        gt:   (H, W) or (N, H, W) torch tensor (gt depth)
    Returns:
        pred at the same resolution as gt
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)            # (1, H', W')
        gt_dim = 2
    else:
        gt_dim = 3
    if gt.dim() == 2:
        gt = gt.unsqueeze(0)
    # Add a channel dim for interpolate
    pred = pred.unsqueeze(1)                 # (N, 1, H', W')
    aligned = F.interpolate(
        pred, size=(gt.shape[-2], gt.shape[-1]),
        mode="bilinear", align_corners=False
    ).squeeze(1)                              # (N, H, W)
    if gt_dim == 2:
        aligned = aligned.squeeze(0)
    return aligned


def _eigen_crop_mask(spatial_shape, device):
    """
    Build a boolean mask matching `spatial_shape` (either (H, W) or
    (N, H, W)) that is True only inside the standard Eigen crop
    [45:471, 41:601] on a 480x640 grid.

    If the resolution is not the native 480x640 (non-standard), the
    crop is skipped (all-True) so metrics degrade gracefully rather
    than silently masking the wrong region.
    """
    crop = torch.zeros(spatial_shape, dtype=torch.bool, device=device)
    H, W = spatial_shape[-2], spatial_shape[-1]
    if (H, W) == (480, 640):
        t, b, l, r = EIGEN_CROP
        crop[..., t:b, l:r] = True
    else:
        crop[...] = True
    return crop


def compute_depth_metrics(
    pred, gt, valid_mask,
    min_depth=NYUV2_MIN_DEPTH, max_depth=NYUV2_MAX_DEPTH,
    use_eigen_crop=True,
):
    """
    Compute the 7 standard depth metrics on a SINGLE image OR on a
    batch (the function handles both — it flattens valid pixels).

    Args:
        pred: predicted depth (same shape as gt or different and will
              be resized to gt). In meters.
        gt:   ground-truth depth in meters. 0 means invalid.
        valid_mask: bool tensor, True where gt is valid.
        min_depth, max_depth: clip range in meters.
        use_eigen_crop: if True (default), restrict evaluation to the
              standard Eigen center crop [45:471, 41:601]. Set False
              to evaluate the full frame.

    Returns:
        dict with keys 'RMSE', 'RMSE_log', 'AbsRel', 'SqRel',
        'delta1', 'delta2', 'delta3', plus 'n_valid'.
    """
    pred = pred.float()
    gt = gt.float()

    # Align pred to gt resolution if needed
    if pred.shape != gt.shape:
        pred = align_pred_to_gt(pred, gt)

    # Clip both to valid depth range
    pred = pred.clamp(min=min_depth, max=max_depth)
    gt   = gt.clamp(  min=min_depth, max=max_depth)

    # Restrict to valid pixels
    mask = valid_mask & (gt > min_depth) & (gt < max_depth)

    # Standard Eigen center crop (applied identically to clean & adv)
    if use_eigen_crop:
        mask = mask & _eigen_crop_mask(mask.shape, mask.device)

    pred_v = pred[mask]
    gt_v   = gt[mask]

    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return {
            "RMSE": float("nan"), "RMSE_log": float("nan"),
            "AbsRel": float("nan"), "SqRel": float("nan"),
            "delta1": float("nan"), "delta2": float("nan"),
            "delta3": float("nan"), "n_valid": 0,
        }

    # Errors
    err = pred_v - gt_v
    rmse = torch.sqrt((err ** 2).mean()).item()
    rmse_log = torch.sqrt(
        ((torch.log(pred_v) - torch.log(gt_v)) ** 2).mean()
    ).item()
    abs_rel = (err.abs() / gt_v).mean().item()
    sq_rel  = ((err ** 2) / gt_v).mean().item()

    # Threshold accuracies (delta-1, delta-2, delta-3)
    ratio = torch.max(pred_v / gt_v, gt_v / pred_v)
    delta1 = (ratio < 1.25     ).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "RMSE": rmse, "RMSE_log": rmse_log,
        "AbsRel": abs_rel, "SqRel": sq_rel,
        "delta1": delta1, "delta2": delta2, "delta3": delta3,
        "n_valid": n_valid,
    }


def average_metrics(metrics_list):
    """
    Average a list of per-image metric dicts into a single summary dict.
    Excludes images that had n_valid == 0.
    """
    keys = ["RMSE", "RMSE_log", "AbsRel", "SqRel", "delta1", "delta2", "delta3"]
    valid = [m for m in metrics_list if m["n_valid"] > 0]
    if not valid:
        return {k: float("nan") for k in keys}
    avg = {}
    for k in keys:
        vals = [m[k] for m in valid]
        avg[k] = float(sum(vals) / len(vals))
    avg["n_images"] = len(valid)
    return avg
