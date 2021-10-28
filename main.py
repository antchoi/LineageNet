"""
Incubator for LineageNet
"""

import sys
import argparse
from datetime import datetime

import torch
from pytorch_lightning import Trainer, plugins
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from pytorch_lightning.plugins import DDPPlugin

import models
import datamodule

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M")

PATH_DATA = "data"
PATH_DATASET = f"{PATH_DATA}/datasets"

reset_seed()
seed_everything(42, workers=True)

NUM_GPUS = torch.cuda.device_count()
# NUM_GPUS = 1
AVAIL_GPUS = list(range(NUM_GPUS)) if NUM_GPUS > 0 else None

MODEL_DICT = {
    "resnet18": models.ResNet18,
    "preactresnet18": models.PreactResNet18,
    "amppreactresnet18": models.AmpPreactResNet18,
}
MODEL_LIST = list(MODEL_DICT.keys())

DATAMODULE_DICT = {
    "cifar10": datamodule.CIFAR10DataModule,
    "cifar100": datamodule.CIFAR100DataModule,
    "ilsvrc2012": datamodule.ILSVRC2012DataModule,
    "ilsvrc2012d64": datamodule.ILSVRC2012D64DataModule,
}
DATASET_LIST = list(DATAMODULE_DICT.keys())

if len(MODEL_LIST) == 0:
    print("No available models")
    sys.exit(1)

parser = argparse.ArgumentParser(description="LineageNet Training")
parser.add_argument(
    "--model",
    "-m",
    default=MODEL_LIST[0],
    type=str,
    help="model type",
    choices=MODEL_LIST,
)

parser.add_argument(
    "--dataset",
    "-d",
    default=DATASET_LIST[0],
    type=str,
    help="dataset type",
    choices=DATASET_LIST,
)

parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)

parser.add_argument(
    "--mixed_precision",
    "-p",
    action="store_true",
    default=True,
    help="use mixed precision",
)

args = parser.parse_args()

MODEL_TYPE = str(args.model).lower()
DATASET_TYPE = str(args.dataset).lower()

if MODEL_TYPE not in MODEL_LIST:
    print(f"Unavailable model: {MODEL_TYPE}")
    sys.exit(1)

if DATASET_TYPE not in DATASET_LIST:
    print(f"Unavailable dataset: {DATASET_TYPE}")
    sys.exit(1)

IS_RESUME = args.resume

PATH_LOG = f"{PATH_DATA}/log/{MODEL_TYPE}/{DATASET_TYPE}/{TIMESTAMP}"
PATH_CHECKPOINT = f"{PATH_LOG}/checkpoints"

MODEL_PRECISION = 32

if __name__ == "__main__":

    datamodule = DATAMODULE_DICT[DATASET_TYPE](
        data_dir=PATH_DATASET,
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#batch-size
        # train_batch_size=7 * NUM_GPUS,
        train_batch_size=7 * NUM_GPUS,
        test_batch_size=100,
        num_workers=4 * NUM_GPUS,
        # num_workers=2,
        image_size=64,
        # pin_memory=False,
    )
    model = MODEL_DICT[MODEL_TYPE](num_classes=datamodule.num_classes)

    logger = TensorBoardLogger(PATH_LOG)

    # saves a file like: my/path/sample-mnist-val_acc=0.12345-epoch=002.ckpt
    checkpoint_acc_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=PATH_CHECKPOINT,
        filename="{val_acc:.6f}-{epoch:03d}",
        save_top_k=1,
        mode="max",
    )

    if args.mixed_precision and not getattr(model, "fixed_precision", False):
        MODEL_PRECISION = 16

    AUTOMATIC_OPTIMIZATION = getattr(model, "automatic_optimization", True)
    ACCELERATION = "ddp"
    plugins = [DDPPlugin(find_unused_parameters=not AUTOMATIC_OPTIMIZATION)]

    trainer = Trainer(
        logger=logger,
        default_root_dir=PATH_DATA,
        gpus=AVAIL_GPUS,
        max_epochs=100,
        progress_bar_refresh_rate=1,
        precision=MODEL_PRECISION,
        accelerator=ACCELERATION,
        plugins=plugins,
        callbacks=[checkpoint_acc_callback],
    )

    trainer.fit(model, datamodule=datamodule)
