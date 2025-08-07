"""
Attention U-Net model implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from config import MODEL_CONFIG

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
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
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, num_groups=num_groups)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, dropout_p=0.2, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + out_channels, out_channels, num_groups=num_groups)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.use_attention = use_attention
        if use_attention:
            self.att = AttentionBlock(F_g=in_channels // 2, F_l=out_channels, F_int=out_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        if self.use_attention:
            x2 = self.att(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class AttentionUNetModel(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = MODEL_CONFIG["input_shape"],
                 num_filters: List[int] = MODEL_CONFIG["num_filters"],
                 bridge_filters: int = MODEL_CONFIG["bridge_filters"],
                 output_channels: int = MODEL_CONFIG["output_channels"],
                 activation: str = MODEL_CONFIG["activation"],
                 num_groups: int = 8,
                 dropout_p: float = 0.2):
        super(AttentionUNetModel, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.bridge_filters = bridge_filters
        self.output_channels = output_channels
        self.activation = activation
        in_channels = input_shape[0]
        self.inc = DoubleConv(in_channels, num_filters[0], num_groups=num_groups)
        self.down1 = Down(num_filters[0], num_filters[1], num_groups=num_groups)
        self.down2 = Down(num_filters[1], num_filters[2], num_groups=num_groups)
        self.down3 = Down(num_filters[2], num_filters[3], num_groups=num_groups)
        self.down4 = Down(num_filters[3], bridge_filters, num_groups=num_groups)
        self.up1 = Up(bridge_filters, num_filters[3], num_groups=num_groups, dropout_p=dropout_p, use_attention=True)
        self.up2 = Up(num_filters[3], num_filters[2], num_groups=num_groups, dropout_p=dropout_p, use_attention=True)
        self.up3 = Up(num_filters[2], num_filters[1], num_groups=num_groups, dropout_p=dropout_p, use_attention=True)
        self.up4 = Up(num_filters[1], num_filters[0], num_groups=num_groups, dropout_p=dropout_p, use_attention=False)
        self.outc = OutConv(num_filters[0], output_channels)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(logits)
        elif self.activation == "softmax":
            return F.softmax(logits, dim=1)
        else:
            return logits
    def summary(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Attention U-Net Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output channels: {self.output_channels}")

def create_attention_unet_model(input_shape: Tuple[int, int, int] = None,
                                **kwargs) -> AttentionUNetModel:
    if input_shape is None:
        input_shape = MODEL_CONFIG["input_shape"]
    return AttentionUNetModel(input_shape=input_shape, **kwargs)
