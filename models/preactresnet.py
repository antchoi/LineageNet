"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics

from pytorch_lightning import LightningModule


class PreactBasicBlock(LightningModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreactBasicBlock, self).__init__()

        self.preact = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.LeakyReLU(inplace=True)
        )

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.preact(x)
        out = self.shortcut(out) + self.model(out)
        return out


class PreactResNet18(LightningModule):
    name: str = "PreactResNet18"

    def __init__(
        self,
        num_classes=10,
    ):
        super().__init__()

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            PreactBasicBlock(64, 64, 1),
            PreactBasicBlock(64, 64, 1),
            PreactBasicBlock(64, 128, 2),
            PreactBasicBlock(128, 128, 1),
            PreactBasicBlock(128, 256, 2),
            PreactBasicBlock(256, 256, 1),
            PreactBasicBlock(256, 512, 2),
            PreactBasicBlock(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


def preact_resnet18():
    return PreactResNet18
