"""CIFAR-10 Dataset

Reference:
[1] Alex Krizhevsky and Geoffrey Hinton
    Learning multiple layers of features from tiny images
"""


from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10

from datamodules import BaseDataModule

from utils.trainer import Stage


class CIFAR10DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = None,
        batch_size_train: int = 64,
        batch_size_val: int = 100,
        batch_size_test: int = 100,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = f"./data/cifar10" if not data_dir else data_dir

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_train = transforms.Compose(
            [
                transforms.Resize(int(self.image_size() * 1.15)),
                transforms.RandomCrop(self.image_size()),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.transform_non_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    @staticmethod
    def num_classes() -> int:
        """
        Return: 10
        """
        return 10

    @staticmethod
    def image_size() -> int:
        """
        Return: 32
        """
        return 32

    def setup(self, stage: Stage = Stage.TRAIN) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage is Stage.TRAIN or not stage:
            self.dataset_train = CIFAR10(
                self.data_dir, train=True, transform=self.transform_train
            )

            self.dataset_val = CIFAR10(
                self.data_dir, train=False, transform=self.transform_non_train
            )

        # Assign test dataset for use in dataloader(s)
        if stage is Stage.TEST:
            self.dataset_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform_non_train
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size_train,
            shuffle=(self.sampler_train is None),
            num_workers=self.num_workers,
            sampler=self.sampler_train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers,
            sampler=self.sampler_val,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            sampler=self.sampler_test,
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def prepare(dataset_dir: str):
        CIFAR10(dataset_dir, train=True, download=True)
        CIFAR10(dataset_dir, train=False, download=True)


def cifar10():
    return CIFAR10DataModule
