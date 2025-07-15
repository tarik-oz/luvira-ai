"""
U-Net model implementation for hair segmentation.
Provides a clean and modular implementation of the U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from model.config import MODEL_CONFIG


class DoubleConv(nn.Module):
    """
    Double convolutional block with group normalization and ReLU activation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, num_groups=num_groups)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv, with dropout after conv.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8, dropout_p: float = 0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + out_channels, out_channels, num_groups=num_groups)
        self.dropout = nn.Dropout2d(p=dropout_p)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class OutConv(nn.Module):
    """
    Output convolution layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    """
    U-Net model for semantic segmentation.
    
    Attributes:
        input_shape: Shape of input images (channels, height, width)
        num_filters: List of filter counts for each encoder/decoder level
        bridge_filters: Number of filters in the bridge layer
        output_channels: Number of output channels
        activation: Activation function for output layer
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = MODEL_CONFIG["input_shape"],
                 num_filters: List[int] = MODEL_CONFIG["num_filters"],
                 bridge_filters: int = MODEL_CONFIG["bridge_filters"],
                 output_channels: int = MODEL_CONFIG["output_channels"],
                 activation: str = MODEL_CONFIG["activation"],
                 num_groups: int = 8,
                 dropout_p: float = 0.2):
        """
        Initialize U-Net model with specified parameters.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            num_filters: List of filter counts for each encoder/decoder level
            bridge_filters: Number of filters in the bridge layer
            output_channels: Number of output channels
            activation: Activation function for output layer
        """
        super(UNetModel, self).__init__()
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.bridge_filters = bridge_filters
        self.output_channels = output_channels
        self.activation = activation
        
        # Input channels
        in_channels = input_shape[0]
        
        # Encoder path
        self.inc = DoubleConv(in_channels, num_filters[0], num_groups=num_groups)
        self.down1 = Down(num_filters[0], num_filters[1], num_groups=num_groups)
        self.down2 = Down(num_filters[1], num_filters[2], num_groups=num_groups)
        self.down3 = Down(num_filters[2], num_filters[3], num_groups=num_groups)
        
        # Bridge
        self.down4 = Down(num_filters[3], bridge_filters, num_groups=num_groups)
        
        # Decoder path
        self.up1 = Up(bridge_filters, num_filters[3], num_groups=num_groups, dropout_p=dropout_p)
        self.up2 = Up(num_filters[3], num_filters[2], num_groups=num_groups, dropout_p=dropout_p)
        self.up3 = Up(num_filters[2], num_filters[1], num_groups=num_groups, dropout_p=dropout_p)
        self.up4 = Up(num_filters[1], num_filters[0], num_groups=num_groups, dropout_p=dropout_p)
        
        # Output layer
        self.outc = OutConv(num_filters[0], output_channels)
        
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        # Apply activation
        if self.activation == "sigmoid":
            return torch.sigmoid(logits)
        elif self.activation == "softmax":
            return F.softmax(logits, dim=1)
        else:
            return logits
    
    def summary(self) -> None:
        """Print model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output channels: {self.output_channels}")


def create_unet_model(input_shape: Tuple[int, int, int] = None,
                     **kwargs) -> UNetModel:
    """
    Factory function to create a U-Net model.
    
    Args:
        input_shape: Shape of input images
        **kwargs: Additional arguments for UNetModel
        
    Returns:
        UNetModel instance
    """
    if input_shape is None:
        input_shape = MODEL_CONFIG["input_shape"]
        
    return UNetModel(input_shape=input_shape, **kwargs)
