"""
Custom loss functions for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification tasks.

    Focal loss down-weights well-classified examples and focuses on hard,
    misclassified examples. This is especially useful for imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        - p_t is the probability of the correct class
        - alpha_t is the class weight (optional)
        - gamma is the focusing parameter (gamma >= 0)

    When gamma = 0, focal loss is equivalent to cross-entropy loss.
    Higher gamma values increase focus on hard examples.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        https://arxiv.org/abs/1708.02002

    Args:
        alpha: Class weights. Can be:
            - None: No class weighting
            - float: Weight for the positive class (binary classification)
            - Tensor: Per-class weights of shape (num_classes,)
        gamma: Focusing parameter. Default is 2.0.
        reduction: Specifies the reduction to apply to the output.
            'none': no reduction
            'mean': mean of the loss
            'sum': sum of the loss
        label_smoothing: Label smoothing factor. Default is 0.0.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Focal loss value
        """
        num_classes = inputs.size(-1)

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probability of true class
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to cross-entropy loss
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(labels: list, num_classes: int, method: str = "inverse") -> torch.Tensor:
    """
    Compute class weights based on label distribution.

    Args:
        labels: List of label indices
        num_classes: Total number of classes
        method: Weighting method
            - "inverse": Inverse of class frequency
            - "inverse_sqrt": Square root of inverse frequency
            - "effective": Effective number of samples (recommended)

    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    import numpy as np
    from collections import Counter

    # Count samples per class
    counts = Counter(labels)
    total = len(labels)

    weights = torch.zeros(num_classes)

    if method == "inverse":
        # Weight inversely proportional to frequency
        for cls_idx in range(num_classes):
            count = counts.get(cls_idx, 1)  # Avoid division by zero
            weights[cls_idx] = total / (num_classes * count)

    elif method == "inverse_sqrt":
        # Square root of inverse frequency (less aggressive)
        for cls_idx in range(num_classes):
            count = counts.get(cls_idx, 1)
            weights[cls_idx] = np.sqrt(total / (num_classes * count))

    elif method == "effective":
        # Effective number of samples (Cui et al., 2019)
        # "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        for cls_idx in range(num_classes):
            count = counts.get(cls_idx, 1)
            effective_num = 1.0 - np.power(beta, count)
            weights[cls_idx] = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes

    return weights
