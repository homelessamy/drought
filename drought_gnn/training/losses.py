"""
MSE and weighted cross-entropy losses.
"""
import torch
import torch.nn as nn

__all__ = ["MSELoss", "weighted_cross_entropy"]

class MSELoss(nn.MSELoss):
    """Mean Squared Error loss for weather state regression."""
    pass

def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """
    Thin wrapper around nn.CrossEntropyLoss with class weights.
    class_weights computed by caller using sklearn.utils.compute_class_weight.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    return criterion(logits, targets)
