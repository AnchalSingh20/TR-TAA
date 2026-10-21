"""
Multi-label classification head.

This is structurally identical to LinearClassificationHead — a single
linear layer on the CLS token — but is named separately so that:
  (a) we don't overwrite the single-label .pth files,
  (b) callers can be explicit about which head they're using.

The architecture difference is purely semantic: train this head with
BCEWithLogitsLoss instead of CrossEntropyLoss, and at evaluation time
pass logits through sigmoid (or threshold at 0) to obtain per-class
binary predictions. We then compute mean Average Precision (mAP)
across the 20 classes, following the TAA evaluation protocol.
"""

import torch
import torch.nn as nn


class MultiLabelClassificationHead(nn.Module):
    """
    Linear head for multi-label classification on top of VFM CLS token.

    - Input: CLS token features (B, D)
    - Output: 20 class logits (B, 20). Apply sigmoid for probabilities.
    """

    def __init__(self, feature_dim, num_classes=20):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        """
        Args:
            features: CLS token (B, D)
        Returns:
            logits: (B, num_classes) — pre-sigmoid scores
        """
        return self.linear(features)


class FullModelMultiLabel(nn.Module):
    """
    Complete multi-label model: VFM backbone + multi-label head.

    Used by Task-Specific Attacks to attack multi-label
    classification directly. For Task-Agnostic Attacks the head is
    not used during generation.
    """

    def __init__(self, backbone, head, normalize_fn=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.normalize_fn = normalize_fn

    def forward(self, x):
        x_norm = self.normalize_fn(x) if self.normalize_fn is not None else x
        features = self.backbone.forward_features(x_norm)
        cls_token = features["x_norm_clstoken"]
        return self.head(cls_token)        # (B, 20) logits
