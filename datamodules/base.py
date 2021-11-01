from abc import abstractmethod, abstractproperty, abstractstaticmethod
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms.transforms import Compose

from utils.trainer import Stage


class BaseDataModule:
    data_dir: str

    image_size: int

    batch_size_train: int
    batch_size_val: int
    batch_size_test: int

    dataset_train: Dataset
    dataset_val: Dataset
    dataset_test: Dataset

    sampler_train: Sampler
    sampler_val: Sampler
    sampler_test: Sampler

    num_workers: int
    pin_memory: bool

    transform_train: Compose
    transform_non_train: Compose

    @abstractstaticmethod
    def num_classes(self) -> int:
        raise NotImplementedError()

    @abstractstaticmethod
    def image_size() -> int:
        raise NotImplementedError()

    @abstractmethod
    def setup(self, _: Stage) -> None:
        raise NotImplementedError()

    def setup_sampler(self, sampler_class: Sampler = None, stage: Stage = Stage.TRAIN):
        if not sampler_class:
            return

        if stage is Stage.TRAIN or not stage:
            self.sampler_train = sampler_class(dataset=self.dataset_train)
            self.sampler_val = sampler_class(dataset=self.dataset_val)

        if stage is Stage.TEST:
            self.sampler_test = sampler_class(dataset=self.dataset_test)

    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError()

    @abstractmethod
    def val_dataloader(self):
        raise NotImplementedError()

    @abstractmethod
    def test_dataloader(self):
        raise NotImplementedError()

    @abstractstaticmethod
    def prepare(_: str):
        raise NotImplementedError()
