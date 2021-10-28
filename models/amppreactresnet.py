"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
import optimizer

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
            # nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False,),
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
        out = self.model(out) + self.shortcut(out)
        return out


class AmpPreactResNet18(LightningModule):
    name: str = "AmpPreactResNet18"

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

        self.automatic_optimization = False
        self.fixed_precision = True

        self.clip_norm = 50

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
            nn.Linear(in_features=2048, out_features=self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        inputs, targets = batch

        def closure():
            opt.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
            return outputs, loss

        opt.step(closure)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        opt = optimizer.AMP(
            self.parameters(),
            lr=self.learning_rate,
            epsilon=0.5,
            inner_iter=1,
            inner_lr=1.0,
            base_optimizer=torch.optim.SGD,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=self.milestones, gamma=0.1
        )

        return {"optimizer": opt, "scheduler": sched, "monitor": "val_loss"}
