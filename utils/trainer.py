from enum import Enum, auto
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torchmetrics

from torch.nn.parallel import DistributedDataParallel as DDP


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()


class GenericTrainer:

    model: nn.Module = None
    criterion: nn.Module = None
    optimizer: torch.optim.Optimizer = None
    lr_scheduler: torch.optim.lr_scheduler = None
    metrics: Dict[str, torchmetrics.Metric] = None
    distributed: bool = None

    metric_top1: torchmetrics.Metric = None
    metric_top5: torchmetrics.Metric = None

    learning_rate: float = 0.1

    train_loss: float = 0.0
    val_loss: float = 0.0

    gpu: int = 0

    def __init__(
        self,
        model: nn.Module,
        learning_rate=0.1,
        gpu: int = 0,
        distributed: bool = True,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.gpu = gpu
        self.distributed = distributed

        self.metric_top1 = torchmetrics.Accuracy(top_k=1)
        self.metric_top5 = torchmetrics.Accuracy(top_k=5)

    def configure(
        self,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
    ):
        self.criterion = nn.CrossEntropyLoss() if not criterion else criterion

        self.optimizer = (
            torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            if not optimizer
            else optimizer
        )

        self.lr_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
            if not lr_scheduler
            else lr_scheduler
        )

    def ready(self):
        self.model.cuda(self.gpu)

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu])

        self.criterion.cuda(self.gpu)
        self.metric_top1.cuda(self.gpu)
        self.metric_top5.cuda(self.gpu)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, labels = batch
        inputs = inputs.cuda(self.gpu, non_blocking=True)
        labels = labels.cuda(self.gpu, non_blocking=True)

        logits = self.model(inputs)
        loss: torch.Tensor = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def on_training_end(self) -> Dict[str, torch.Tensor]:
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, labels = batch
        inputs = inputs.cuda(self.gpu, non_blocking=True)
        labels = labels.cuda(self.gpu, non_blocking=True)

        logits = self.model(inputs)
        loss = self.criterion(logits, labels)

        self.metric_top1.update(logits, labels)
        self.metric_top5.update(logits, labels)

        return loss

    def on_validation_end(self):
        param = None
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            param = self.val_loss

        self.lr_scheduler.step(param)
        self.val_loss = 0.0

    def compute_metric_top1(self, keep: bool = False) -> torch.Tensor:
        val = self.metric_top1.compute()
        if not keep:
            self.metric_top1.reset()
        return val

    def compute_metric_top5(self, keep: bool = False) -> torch.Tensor:
        val = self.metric_top5.compute()
        if not keep:
            self.metric_top5.reset()
        return val

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
