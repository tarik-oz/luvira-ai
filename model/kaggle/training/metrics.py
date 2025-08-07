"""
Metrics calculation functions for hair segmentation training.
"""

import torch

from config import DATA_CONFIG

DICE_SMOOTHING = 1e-6

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy for binary segmentation."""
    with torch.no_grad():
        # Convert predictions and targets to binary using config threshold
        threshold = DATA_CONFIG["mask_threshold"]
        binary_preds = (predictions > threshold).float()
        binary_targets = (targets > threshold).float()
        # Calculate accuracy
        accuracy = (binary_preds == binary_targets).float().mean().item()
    return accuracy


def calculate_dice(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate Dice coefficient for probability masks."""
    with torch.no_grad():
        preds = predictions.view(-1)
        targs = targets.view(-1)
        intersection = (preds * targs).sum()
        dice = (2. * intersection + DICE_SMOOTHING) / (preds.sum() + targs.sum() + DICE_SMOOTHING)
    return dice.item()


def calculate_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate Mean Squared Error for probability masks."""
    with torch.no_grad():
        mse = torch.mean((predictions - targets) ** 2)
    return mse.item() 