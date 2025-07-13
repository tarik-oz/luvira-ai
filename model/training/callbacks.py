"""
Callbacks and loss functions for hair segmentation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# Constants
BEST_LOSS_INIT = float('inf')
WORST_ACCURACY_INIT = float('-inf')


class EarlyStopping:
    """Early stopping callback for PyTorch training."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_metric = BEST_LOSS_INIT if monitor.endswith('loss') else WORST_ACCURACY_INIT
        
    def __call__(self, metric_value: float) -> bool:
        # For loss metrics: lower is better
        # For accuracy/dice metrics: higher is better
        if self.monitor.endswith('loss'):
            is_better = metric_value < self.best_metric - self.min_delta
        else:
            is_better = metric_value > self.best_metric + self.min_delta
            
        if is_better:
            self.best_metric = metric_value
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        return self.bce_weight * self.bce(inputs, targets) + self.dice_weight * self.dice(inputs, targets)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs, targets):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=inputs.dtype, device=inputs.device).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=inputs.dtype, device=inputs.device).unsqueeze(0).unsqueeze(0) / 8.0
        edge_pred = F.conv2d(inputs, sobel_x, padding=1) + F.conv2d(inputs, sobel_y, padding=1)
        edge_true = F.conv2d(targets, sobel_x, padding=1) + F.conv2d(targets, sobel_y, padding=1)
        return F.l1_loss(edge_pred, edge_true)


class TotalLoss(nn.Module):
    def __init__(self, bce_weight=0.4, dice_weight=0.4, boundary_weight=0.2):
        super(TotalLoss, self).__init__()
        self.combo = ComboLoss(bce_weight, dice_weight)
        self.boundary = BoundaryLoss()
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets):
        return self.combo(inputs, targets) + self.boundary_weight * self.boundary(inputs, targets) 