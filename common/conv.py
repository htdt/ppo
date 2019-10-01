from typing import List, Tuple
import torch
from common.tools import init_ortho, Flatten
import torch.nn as nn


class Conv(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int],
            channels: List[int],
            kernel: List[int],
            stride: List[int],
    ):
        super(Conv, self).__init__()
        assert len(channels) == len(kernel) == len(stride)

        def with_relu(m):
            return nn.Sequential(init_ortho(m, 'relu'), nn.ReLU())
        params = zip([input_size[0]] + channels[:-1], channels, kernel, stride)
        self.conv = nn.Sequential(*[with_relu(nn.Conv2d(*p)) for p in params],
                                  Flatten())

        with torch.no_grad():
            tmp = torch.zeros((1,) + input_size)
            self.output_size = self.conv(tmp).shape[-1]

    def forward(self, x):
        return self.conv(x.float() / 255)
