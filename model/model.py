import torch
from torch import nn
from torch.nn import functional as F

from model.blocks import ResBlock, ResPoolBlock
from hparameters import data_config

class CNNet(nn.Module):
  def __init__(
      self,
      num_layers,
      initial_channels,
      feat_size
    ):
    super(CNNet, self).__init__()

    self.model = nn.Sequential()

    self.model.append(
        nn.Conv2d(
            in_channels=data_config.IMAGE_CHANNELS,
            out_channels=initial_channels,
            kernel_size=3,
            stride=1,
            padding='same'
        )
    )

    for _ in range(num_layers):
      self.model.append(
          ResBlock(
              channels=initial_channels,
              feat_size=feat_size
          )
      )

      feat_size = feat_size // 2

      self.model.append(
          ResPoolBlock(
              in_channels=initial_channels,
              out_channels=initial_channels * 2,
              feat_size=feat_size
          )
      )

      initial_channels = initial_channels * 2

    feat_size = feat_size // 2

    self.model.append(
        nn.AdaptiveAvgPool2d(
            (feat_size, feat_size)
        )
    )

    self.last_layer = nn.Linear(
        in_features=feat_size * feat_size * initial_channels,
        out_features=data_config.NUM_CLASSES
    )

  def forward(self, inputs):
    x = self.model(inputs)
    x = torch.flatten(x, 1)
    x = self.last_layer(x)

    return x

