"""
Multi-label metrics: mean Average Precision (mAP) for PascalVOC.

This follows the TAA evaluation protocol (Pulfer et al., CVPRW 2025).
Clean mAP on PascalVOC val with DINOv2 ViT-S/14 is typically ~95-96%.

mAP is computed as the mean over 20 classes of per-class Average Precision (AP),
where AP is the area under the precision-recall curve.

We use sklearn.metrics.average_precision_score for the per-class AP — it is
the standard implementation matching VOC/PR competition conventions.
"""

import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from src.models import normalize_for_model


# ============================================================
# Full-dataset evaluation (used by train script)
# ============================================================

def evaluate_classification_multilabel(backbone, head, dataloader, device,
                                       desc="Eval"):
    """
    Evaluate multi-label classification mAP on a DataLoader.

    Args:
        backbone: VFM backbone (frozen)
        head: MultiLabelClassificationHead
        dataloader: yields (images, multi_hot_labels, paths)
        device: torch device
        desc: tqdm description

    Returns:
        mAP in percent (float in [0, 100])
    """
    backbone.eval()
    head.eval()

    all_scores = []        # list of (B, 20) score arrays
    all_labels = []        # list of (B, 20) label arrays

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)            # (B, 20) float

            images_norm = normalize_for_model(images)
            features = backbone.forward_features(images_norm)
            cls_token = features["x_norm_clstoken"]

            logits = head(cls_token)              # (B, 20)
            scores = torch.sigmoid(logits)        # (B, 20) probabilities

            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    scores_arr = np.concatenate(all_scores, axis=0)   # (N, 20)
    labels_arr = np.concatenate(all_labels, axis=0)   # (N, 20)

    return compute_map(scores_arr, labels_arr)


def compute_map(scores, labels):
    """
    Compute mean Average Precision over classes.

    Args:
        scores: (N, num_classes) prediction scores in [0, 1]
        labels: (N, num_classes) binary ground-truth labels

    Returns:
        mAP in percent (float in [0, 100])
    """
    num_classes = scores.shape[1]
    aps = []
    for c in range(num_classes):
        if labels[:, c].sum() == 0:
            # No positives for this class in this slice: skip
            continue
        ap = average_precision_score(labels[:, c], scores[:, c])
        aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps)) * 100.0


# ============================================================
# Per-batch evaluation (used by adversarial-eval scripts)
#
# Pattern matches evaluate_classification_on_adversarial in metrics.py.
# Instead of returning (correct, total) we accumulate (scores, labels)
# arrays externally and compute mAP at the end. This matches sklearn's
# requirement that AP be computed over the full validation set, not
# averaged across batches.
# ============================================================

def collect_scores_and_labels_on_adversarial(backbone, head,
                                             adv_images, labels, device):
    """
    Run a single batch of adversarial images through backbone+head and
    return (scores, labels) as numpy arrays so the caller can accumulate
    them and compute mAP at the end.

    Args:
        backbone: VFM backbone
        head: MultiLabelClassificationHead
        adv_images: (B, 3, H, W) in [0, 1]
        labels: (B, 20) multi-hot tensor
        device: torch device

    Returns:
        scores_np: (B, 20) numpy array of sigmoid probabilities
        labels_np: (B, 20) numpy array of ground-truth labels
    """
    backbone.eval()
    head.eval()

    with torch.no_grad():
        adv_images = adv_images.to(device)
        labels = labels.to(device)

        images_norm = normalize_for_model(adv_images)
        features = backbone.forward_features(images_norm)
        cls_token = features["x_norm_clstoken"]

        logits = head(cls_token)
        scores = torch.sigmoid(logits)

    return scores.cpu().numpy(), labels.cpu().numpy()
