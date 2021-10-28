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

# from deepspeed.ops.adam import FusedAdam


class BasicBlock(LightningModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

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
            nn.BatchNorm2d(out_channels),
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
        out = self.model(x)
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class ResNet18(LightningModule):
    name: str = "ResNet18"

    def __init__(
        self,
        learning_rate=0.1,
        criterion=nn.CrossEntropyLoss(),
        metric=torchmetrics.Accuracy(),
        num_classes=10,
        max_epoch=500,
        milestones=[100, 200, 300, 400],
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.milestones = milestones

        self.criterion = criterion
        self.metric = metric

        # Hardcode some dataset specific attributes
        self.num_classes = num_classes

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # Input: [3, 224, 224]
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
            # Output: [64, 112, 112]
            # Input: [64, 112, 112]
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1),
            # Output: [128, 56, 56]
            # Input: [128, 56, 56]
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1),
            # Output: [256, 28, 28]
            # Input: [256, 28, 28]
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1),
            # Output: [512, 14, 14]
            # nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(
            "val_loss", loss, prog_bar=True, logger=True,
        )
        self.log("val_acc", acc, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        # optimizer = FusedAdam(self.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=0.1
        )

        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}
