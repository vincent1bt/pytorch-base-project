import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, feat_size):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                padding='same',
                kernel_size=3,
                stride=1,
            ),
            nn.LayerNorm([channels, feat_size, feat_size]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                padding='same',
                kernel_size=3,
                stride=1
            ),
            nn.LayerNorm([channels, feat_size, feat_size]),
        )

    def forward(self, inputs):
        return F.silu(self.block(inputs) + inputs)

class ResPoolBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            feat_size,
        ):
        super(ResPoolBlock, self).__init__()

        stride = 2
        kernel_size = 3
        dilation = 1
        self.padding = ((stride - 1) * dilation + kernel_size - 1) // 2 # 1

        self.pool = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=0,
                stride=stride,
                kernel_size=kernel_size
            ),
            nn.LayerNorm([out_channels, feat_size, feat_size]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                padding='same',
                stride=1,
                kernel_size=kernel_size,
            ),
            nn.LayerNorm([out_channels, feat_size, feat_size]),
        )

    def forward(self, inputs):
        inputs_padded = F.pad(
            inputs,
            [self.padding, self.padding, self.padding, self.padding],
            value=0
        )

        return F.silu(
            self.block(inputs_padded) + self.pool(inputs)
        )

