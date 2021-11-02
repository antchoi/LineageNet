"""
Cerberus Network
"""

import os
import shutil
import argparse
from datetime import datetime

import abc
from tqdm import tqdm

from multiprocessing.managers import SyncManager


import torch
from torch import nn
from torch.backends import cudnn

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

from torchinfo import summary

from pytorch_lightning.utilities.seed import reset_seed, seed_everything

import models as Models
import datamodules as DataModules

from datamodules import BaseDataModule

from utils import GenericTrainer
from utils.trainer import Stage

MODEL_NAMES = sorted(
    name
    for name in Models.__dict__
    if name.islower() and not name.startswith("__") and callable(Models.__dict__[name])
)

DATASET_NAMES = sorted(
    name
    for name in DataModules.__dict__
    if name.islower()
    and not name.startswith("__")
    and not name in abc.__dict__
    and callable(DataModules.__dict__[name])
)

parser = argparse.ArgumentParser(description="LineageNet Training")

parser.add_argument(
    "--dataset",
    "-d",
    default=DATASET_NAMES[0],
    choices=DATASET_NAMES,
    help="datasets: " + " | ".join(DATASET_NAMES) + f" (default: {DATASET_NAMES[0]})",
)

parser.add_argument(
    "--dataset-dir",
    metavar="DIR",
    default=None,
    help="path to dataset",
    dest="dataset_dir",
)

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=MODEL_NAMES,
    help="model architecture: " + " | ".join(MODEL_NAMES) + " (default: resnet18)",
)

parser.add_argument(
    "--epochs",
    default=90,
    type=int,
    metavar="N",
    help="number of total epochs to run",
    dest="epochs",
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)

parser.add_argument(
    "-b",
    "--batch-size-train",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256) for training, this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
    dest="batch_size_train",
)

parser.add_argument(
    "--bv",
    "--batch-size-val",
    default=100,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256) for validation, this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
    dest="batch_size_val",
)

parser.add_argument(
    "--bt",
    "--batch-size-test",
    default=100,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256) for testing, this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
    dest="batch_size_test",
)

parser.add_argument(
    "--checkpoint",
    "-c",
    type=str,
    help="resume from the givin checkpoint",
    dest="checkpoint",
)

parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)

parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)

parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

parser.add_argument(
    "--world-size",
    default=1,
    type=int,
    metavar="N",
    help="number of nodes for distributed training",
)

parser.add_argument(
    "--seed",
    default=None,
    type=int,
    metavar="N",
    help="seed for initializing training. ",
)


def load_checkpoint(path: str):
    assert os.path.isfile(path), f"Error: checkpoint file not found: {path}"

    # net, val/acc, val/loss, last_epoch
    return torch.load(path)


def cleanup():
    dist.destroy_process_group()


def select_model_class(model_type: str) -> nn.Module:
    return Models.__dict__[model_type]()


def select_datamodule_class(dataset_type: str) -> nn.Module:
    return DataModules.__dict__[dataset_type]()


def print_model_summary(
    model: nn.Module, batch_size: int, image_size: int, device: str = "cpu"
):
    summary(
        model,
        (batch_size, 3, image_size, image_size),
        dtypes=[torch.float],
        device=device,
    )


def save_checkpoint(
    state: object,
    is_best: bool = False,
    checkpoint_dir: str = "./",
    filename="checkpoint.pth.tar",
):
    filepath = f"{checkpoint_dir}/{filename}"
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, f"{checkpoint_dir}/model_best.pth.tar")


# ------ Setting up the distributed environment -------
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.


def main():
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    data_dir = "./data"
    args.log_dir = f"{data_dir}/log/{args.arch}/{args.dataset}/{timestamp}"
    args.checkpoint_dir = f"{args.log_dir}/checkpoint"

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.dataset_dir is None:
        args.dataset_dir = f"{data_dir}/datasets/{args.dataset}"

    datamodule_class = select_datamodule_class(args.dataset)
    datamodule_class.prepare(args.dataset_dir)

    model_class = select_model_class(args.arch)
    model: nn.Module = model_class(datamodule_class.num_classes())

    print_model_summary(
        model, args.batch_size_train, datamodule_class.image_size(), device="cpu"
    )

    reset_seed()

    if args.seed is not None:
        seed_everything(42, workers=True)
        cudnn.deterministic = True

    ngpus_per_node = torch.cuda.device_count()

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size

    mp.freeze_support()

    manager: SyncManager = mp.Manager()
    write_lock = manager.Lock()

    with mp.Manager() as manager:
        # main_worker process function
        mp.spawn(
            train_model,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, write_lock),
            join=True,
        )


def train_model(
    gpu: int, ngpus_per_node: int, args: argparse.Namespace, write_lock: mp.Lock = None
):
    print(f"Running Distributed {args.arch} on gpu {gpu}.")

    tqdm.set_lock(write_lock)

    cudnn.benchmark = True

    args.gpu = gpu

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu

    setup(args.rank, args.world_size)
    torch.cuda.set_device(args.gpu)

    main_worker = gpu == 0

    if main_worker:
        logger = SummaryWriter(args.log_dir)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size_train = int(args.batch_size_train / ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    datamodule_class = select_datamodule_class(args.dataset)
    datamodule: BaseDataModule = datamodule_class(
        data_dir=args.dataset_dir,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_train,
        num_workers=args.workers,
        pin_memory=True,
    )

    datamodule.setup(stage=Stage.TRAIN)
    datamodule.setup_sampler(DistributedSampler, Stage.TRAIN)

    model_class = select_model_class(args.arch)

    trainer = GenericTrainer(
        model_class(num_classes=datamodule.num_classes()),
        learning_rate=args.lr,
        gpu=args.gpu,
        distributed=True,
    )

    trainer.configure()
    trainer.ready()

    best_acc_top1 = 0.0
    best_acc_top5 = 0.0

    # Training
    for epoch in range(args.epochs):
        datamodule.sampler_train.set_epoch(epoch)
        datamodule.sampler_val.set_epoch(epoch)

        trainer.model.train()
        loss_epoch = torch.tensor(0.0).to(args.gpu, non_blocking=True)

        pbar_train = tqdm(
            datamodule.train_dataloader(),
            position=args.gpu,
            leave=False,
            desc=f"[{args.gpu}] Epoch {epoch:3d}",
        )

        n_batch = 0
        for idx, batch in enumerate(pbar_train):
            n_batch += 1
            loss_epoch += trainer.training_step(batch, idx)

            postfix_str = "(Step Loss: {:.3f} | Best Top-1 Acc: {:.3f} | Best Top-5 Acc: {:.3f})".format(
                loss_epoch.item() / float(idx + 1), best_acc_top1, best_acc_top5
            )
            pbar_train.set_postfix_str(postfix_str)
            pbar_train.refresh()

        avg_loss_epoch = loss_epoch.item() / float(n_batch)

        if main_worker:
            logger.add_scalar("Train Loss (epoch)", avg_loss_epoch, epoch)

        trainer.on_training_end()
        pbar_train.clear()

        # Validation
        with torch.no_grad():
            loss_epoch = torch.tensor(0.0).to(args.gpu, non_blocking=True)

            pbar_val = tqdm(
                datamodule.val_dataloader(),
                position=args.gpu,
                leave=False,
                desc=f"[{args.gpu}] Epoch {epoch:3d}",
            )
            for idx, batch in enumerate(pbar_val):
                loss_epoch += trainer.validation_step(batch, idx)

                postfix_str = f"(Step Loss: {loss_epoch.item() / float(idx + 1):.3f})"
                pbar_val.set_postfix_str(postfix_str)
                pbar_val.refresh()

            trainer.on_validation_end()
            pbar_val.clear()

            avg_loss_epoch = loss_epoch.item() / float(len(pbar_val))
            acc_top1_epoch = trainer.compute_metric_top1().item()
            acc_top5_epoch = trainer.compute_metric_top5().item()

            # this condition ensures that processes do not trample each other and corrupt the files by overwriting
            if main_worker:
                output = f"[Epoch {epoch:3d}] Validation Loss: {avg_loss_epoch:.3f}"
                output += f", Top-1 Accuracy: {acc_top1_epoch:.3f}, Top-5 Accuracy: {acc_top5_epoch:.3f}"

                logger.add_scalar("Top-1 Accuracy", acc_top1_epoch, epoch)
                logger.add_scalar("Top-5 Accuracy", acc_top5_epoch, epoch)

                state = {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": trainer.model.state_dict(),
                    "best_acc1": acc_top1_epoch,
                    "best_acc5": acc_top5_epoch,
                }

                save_checkpoint(
                    state,
                    is_best=acc_top1_epoch > best_acc_top1,
                    checkpoint_dir=args.checkpoint_dir,
                    filename=f"epoch={epoch}_acc={acc_top1_epoch:.3f}.pth.tar",
                )

                best_acc_top1 = max(best_acc_top1, acc_top1_epoch)
                best_acc_top5 = max(best_acc_top5, acc_top5_epoch)

    cleanup()


if __name__ == "__main__":
    main()
