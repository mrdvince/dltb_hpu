import os

import albumentations as A
import hydra
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, random_split

from datasets import UNETDataset
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
