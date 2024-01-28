import sys

sys.path.append("../")

from torch import nn
import torch
import ipdb
from torchinfo import summary
from constants import *


class Generator(nn.Module):
    def __init__(self, im_chan=1, z_dim=10, hidden_dim=64):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        self.generator = nn.Sequential(
            self._generator_block(
                input_channels=z_dim,
                output_channels=hidden_dim * 4,
            ),  # [B, 1, 1, 1] -> [B, 256, 3, 3]
            self._generator_block(
                input_channels=hidden_dim * 4,
                output_channels=hidden_dim * 2,
                kernel_size=4,
                stride=1,
            ),  # [B, 256, 3, 3] -> [B, 128, 6, 6]
            self._generator_block(
                input_channels=hidden_dim * 2,
                output_channels=hidden_dim,
            ),  # [B, 128, 6, 6] -> [B, 64, 13, 13]
            self._generator_block(
                input_channels=hidden_dim,
                output_channels=im_chan,
                kernel_size=4,
                final_layer=True,
            ),  # [B, 64, 13, 13] -> [B, 1, 28, 28]
        )

    def _generator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        # ipdb.set_trace()
        x = noise.view(len(noise), self.z_dim, 1, 1)  # 2D -> 4D
        return self.generator(x)


def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, device=device)


if __name__ == "__main__":
    ipdb.set_trace()

    test_noise = get_noise(10, Z_DIM)
    gen = Generator(im_chan=1, z_dim=Z_DIM, hidden_dim=64)

    print(summary(gen, input_data=test_noise, verbose=2))
    """
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Generator                                [10, 1, 28, 28]           --
    ├─Sequential: 1-1                        [10, 1, 28, 28]           --
    │    └─Sequential: 2-1                   [10, 256, 3, 3]           --
    │    │    └─ConvTranspose2d: 3-1         [10, 256, 3, 3]           230,656
    │    │    └─BatchNorm2d: 3-2             [10, 256, 3, 3]           512
    │    │    └─ReLU: 3-3                    [10, 256, 3, 3]           --
    │    └─Sequential: 2-2                   [10, 128, 6, 6]           --
    │    │    └─ConvTranspose2d: 3-4         [10, 128, 6, 6]           524,416
    │    │    └─BatchNorm2d: 3-5             [10, 128, 6, 6]           256
    │    │    └─ReLU: 3-6                    [10, 128, 6, 6]           --
    │    └─Sequential: 2-3                   [10, 64, 13, 13]          --
    │    │    └─ConvTranspose2d: 3-7         [10, 64, 13, 13]          73,792
    │    │    └─BatchNorm2d: 3-8             [10, 64, 13, 13]          128
    │    │    └─ReLU: 3-9                    [10, 64, 13, 13]          --
    │    └─Sequential: 2-4                   [10, 1, 28, 28]           --
    │    │    └─ConvTranspose2d: 3-10        [10, 1, 28, 28]           1,025
    │    │    └─Tanh: 3-11                   [10, 1, 28, 28]           --
    ==========================================================================================
    """
