"""ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)

Reference:
[1] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, 
    Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein,
    Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014. 
"""


from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets.imagenet import ImageNet

from datamodules import BaseDataModule

from utils.trainer import Stage


class ILSVRC2012D64DataModule(BaseDataModule):
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
        self.data_dir = f"./data/ilsvrc2012" if not data_dir else data_dir

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size()),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform_non_train = transforms.Compose(
            [
                transforms.Resize(int(self.image_size() * 1.1)),
                transforms.CenterCrop(self.image_size()),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def num_classes() -> int:
        """
        Return: 1000
        """
        return 1000

    @staticmethod
    def image_size() -> int:
        """
        Return: 64
        """
        return 64

    def setup(self, stage: Stage = Stage.TRAIN) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage is Stage.TRAIN or not stage:
            self.dataset_train = ImageNet(
                self.data_dir, split="train", transform=self.transform_train
            )

            self.dataset_val = ImageNet(
                self.data_dir, split="val", transform=self.transform_non_train
            )

        # Assign test dataset for use in dataloader(s)
        if stage is Stage.TEST:
            self.dataset_test = ImageNet(
                self.data_dir, split="val", transform=self.transform_non_train
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
    def prepare(_: str):
        pass


def ilsvrc2012d64():
    return ILSVRC2012D64DataModule
