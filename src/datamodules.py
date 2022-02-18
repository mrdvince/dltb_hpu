import os
from typing import Optional

import albumentations as A
import hydra
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from datasets import CLSDataset, UNETDataset
from utils import get_data


class UNETDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(UNETDataModule, self).__init__()
        self.project_root = hydra.utils.get_original_cwd() + "/"
        self.config = config
        dim = config.data.lung_mask_dim
        self.transforms = A.Compose(
            [
                A.Resize(height=dim, width=dim, always_apply=True),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        self.cxr_dir = self.project_root + config.data.cxr_dir
        self.mask_dir = self.project_root + config.data.mask_dir
        self.bs = config.data.lm_batch_size

    def prepare_data(self):
        if not os.path.exists(self.project_root + self.config.data.lung_mask_raw_dir):
            get_data(
                cxr_dir=self.cxr_dir,
                mask_dir=self.mask_dir,
                data_dir=self.project_root + "data/",
                raw_image_dir=self.project_root + self.config.data.lung_mask_raw_dir,
            )

    def setup(self, stage=None):
        dataset = UNETDataset(
            cxr_dir=self.cxr_dir, mask_dir=self.mask_dir, transform=self.transforms
        )
        train_samples = int(len(dataset) * 0.8)
        self.train_data, self.val_data = random_split(
            dataset, [train_samples, len(dataset) - train_samples]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.bs,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.bs,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(ClassifierDataModule, self).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.config = config
        self.project_root = hydra.utils.get_original_cwd() + "/"
        self.data_dir = self.project_root + self.config.data.data_dir + "/"
        self.cxr_dir = self.project_root + config.data.cxr_dir

    def prepare_data(self) -> None:
        if not os.path.exists(self.project_root + self.config.data.lung_mask_raw_dir):
            get_data()

    def setup(self, stage: Optional[str] = None) -> None:
        # load data
        dataset = CLSDataset(
            cxr_dir=self.cxr_dir, transforms=self.transforms
        )
        train_samples = int(len(dataset) * 0.8)
        self.train_data, self.val_data = random_split(
            dataset, [train_samples, len(dataset) - train_samples]
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.config.data.cl_batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.config.data.cl_batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    train_loader = UNETDataModule(cfg)
    train_loader.prepare_data()
    train_loader.setup()
    train_loader = train_loader.train_dataloader()
    data = next(iter(train_loader))
    loader_images, loader_masks = data
    print(loader_images.shape, loader_masks.shape)


if __name__ == "__main__":
    main()
