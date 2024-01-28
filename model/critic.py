from torch import nn
import torch
import ipdb
from torchinfo import summary


class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            self._critic_block(im_chan, hidden_dim),  # [B, 1, 28, 28] -> [B, 64, 13, 13]
            self._critic_block(hidden_dim, hidden_dim * 2),  # [B, 64, 13, 13] -> [B, 128, 5, 5] (floor[5.5])
            self._critic_block(hidden_dim * 2, 1, final_layer=True),  # [B, 128, 5, 5] -> [B, 1, 1, 1] (floor[1.5])
        )

    def _critic_block(
        self,
        input_channels,
        out_channels,
        kernel_size=4,
        stride=2,  # strided convolutions
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size, stride),
            )

    def forward(self, image):
        critic_pred = self.critic(image)
        return critic_pred.reshape(len(critic_pred), -1)


if __name__ == "__main__":
    ipdb.set_trace()
    test_input = torch.randn(10, 1, 28, 28)
    critic = Critic(im_chan=1, hidden_dim=64)
    test_output = critic(test_input)
    print(f"{test_output.shape=}")
    print(summary(critic, input_data=test_input, verbose=1))

    """

    Critic(
        (critic): Sequential(
            (0): Sequential(
            (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.2, inplace=True)
            )
            (1): Sequential(
            (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2))
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.2, inplace=True)
            )
            (2): Sequential(
            (0): Conv2d(128, 1, kernel_size=(4, 4), stride=(2, 2))
            )
        )
    )

    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Critic                                   [10, 1]                   --
    ├─Sequential: 1-1                        [10, 1, 1, 1]             --
    │    └─Sequential: 2-1                   [10, 64, 13, 13]          --
    │    │    └─Conv2d: 3-1                  [10, 64, 13, 13]          1,088
    │    │    └─BatchNorm2d: 3-2             [10, 64, 13, 13]          128
    │    │    └─LeakyReLU: 3-3               [10, 64, 13, 13]          --
    │    └─Sequential: 2-2                   [10, 128, 5, 5]           --
    │    │    └─Conv2d: 3-4                  [10, 128, 5, 5]           131,200
    │    │    └─BatchNorm2d: 3-5             [10, 128, 5, 5]           256
    │    │    └─LeakyReLU: 3-6               [10, 128, 5, 5]           --
    │    └─Sequential: 2-3                   [10, 1, 1, 1]             --
    │    │    └─Conv2d: 3-7                  [10, 1, 1, 1]             2,049
    ==========================================================================================
    """
