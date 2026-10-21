"""
Evaluation metrics for downstream tasks.

Classification: Accuracy
Segmentation: Mean Intersection over Union (mIoU)

The segmentation confusion matrix is computed in vectorised form.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.models import normalize_for_model


def _update_confusion_matrix(confusion, preds, targets, num_classes):
    """
    Vectorized confusion matrix update — NO Python loops.

    Uses the standard trick: linearize (target, pred) pairs into a 1D index,
    then use bincount to accumulate.

    Args:
        confusion: (num_classes, num_classes) tensor to update in-place
        preds: (N,) predicted class indices
        targets: (N,) ground truth class indices
        num_classes: Number of classes
    """
    valid = (targets >= 0) & (targets < num_classes) & \
            (preds >= 0) & (preds < num_classes)
    t = targets[valid]
    p = preds[valid]

    indices = t * num_classes + p
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes)


def evaluate_classification(backbone, head, dataloader, device, desc="Eval"):
    """Evaluate classification accuracy."""
    backbone.eval()
    head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)

            images_norm = normalize_for_model(images)
            features = backbone.forward_features(images_norm)
            cls_token = features["x_norm_clstoken"]

            logits = head(cls_token)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.shape[0]

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_segmentation(backbone, head, dataloader, device, num_classes=21,
                          desc="Eval"):
    """Evaluate segmentation using mean Intersection over Union (mIoU)."""
    backbone.eval()
    head.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc=desc):
            images = images.to(device)
            masks = masks.to(device)

            images_norm = normalize_for_model(images)
            features = backbone.forward_features(images_norm)
            cls_token = features["x_norm_clstoken"]
            patch_tokens = features["x_norm_patchtokens"]

            logits = head(cls_token, patch_tokens)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            preds = logits.argmax(dim=1)  # (B, H, W)

            # Vectorized confusion matrix update
            valid = masks != -1
            _update_confusion_matrix(
                confusion,
                preds[valid].view(-1),
                masks[valid].view(-1),
                num_classes,
            )

    miou = confusion_to_miou(confusion, num_classes)
    return miou


def evaluate_classification_on_adversarial(backbone, head, adv_images, labels, device):
    """Evaluate classification on a single batch of adversarial images."""
    backbone.eval()
    head.eval()

    with torch.no_grad():
        adv_images = adv_images.to(device)
        labels = labels.to(device)

        images_norm = normalize_for_model(adv_images)
        features = backbone.forward_features(images_norm)
        cls_token = features["x_norm_clstoken"]

        logits = head(cls_token)
        preds = logits.argmax(dim=-1)

        correct = (preds == labels).sum().item()
        total = labels.shape[0]

    return correct, total


def evaluate_segmentation_on_adversarial(backbone, head, adv_images, masks,
                                          device, num_classes=21):
    """Evaluate segmentation on a single batch of adversarial images."""
    backbone.eval()
    head.eval()

    with torch.no_grad():
        adv_images = adv_images.to(device)
        masks = masks.to(device)

        images_norm = normalize_for_model(adv_images)
        features = backbone.forward_features(images_norm)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]

        logits = head(cls_token, patch_tokens)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)

        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long,
                                device=device)
        valid = masks != -1
        _update_confusion_matrix(
            confusion,
            preds[valid].view(-1),
            masks[valid].view(-1),
            num_classes,
        )

    return confusion


def confusion_to_miou(confusion, num_classes=21):
    """Compute mIoU from accumulated confusion matrix."""
    intersection = confusion.diag()
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - intersection
    valid_classes = union > 0
    iou = intersection[valid_classes].float() / union[valid_classes].float()
    return iou.mean().item() * 100.0