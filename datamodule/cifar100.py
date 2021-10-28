"""CIFAR-100 Dataset

Reference:
[1] Alex Krizhevsky and Geoffrey Hinton
    Learning multiple layers of features from tiny images
"""


from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningDataModule

from torchvision import transforms
from torchvision.datasets import CIFAR100


class CIFAR100DataModule(LightningDataModule):
    name: str = "CIFAR100"

    def __init__(
        self,
        data_dir: str = "./data",
        train_batch_size: int = 64,
        test_batch_size: int = 100,
        num_workers: int = 4,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = f"{data_dir}/cifar100"
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

    @property
    def num_classes(self) -> int:
        """
        Return:

            100

        """
        return 100

    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dataset_train = CIFAR100(
                self.data_dir, train=True, transform=self.transform_train
            )

            self.dataset_val = CIFAR100(
                self.data_dir, train=False, transform=self.transform_test
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = CIFAR100(
                self.data_dir, train=False, transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

