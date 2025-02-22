# based on https://github.com/luuuyi/CBAM.PyTorch

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction, dropout):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(num_channels // reduction, num_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((avg_out, max_out), dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(
        self,
        ca_num_channels: int,
        ca_reduction: int,
        sa_kernel_size: int,
        dropout: float = 0.0,
        retain_output: bool = False,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            ca_num_channels, ca_reduction, dropout
        )
        self.spatial_attention = SpatialAttention(sa_kernel_size)

        self.retain_output = retain_output

    def forward(self, x):
        ca_out = self.channel_attention(x)
        x = ca_out * x
        sa_out = self.spatial_attention(x)
        x = sa_out * x

        if self.retain_output:
            self.ca_out = ca_out
            self.sa_out = sa_out

        return x
