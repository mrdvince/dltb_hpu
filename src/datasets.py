import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

png = ".png"


class UNETDataset:
    def __init__(self, cxr_dir, mask_dir, transform=None):
        self.cxr_images = glob(os.path.join(cxr_dir, "*{}".format(png)))
        self.mask_images = glob(os.path.join(mask_dir, "*{}".format(png)))
        self.transform = transform

    def __len__(self):
        return len(self.cxr_images)

    def __getitem__(self, idx):
        cxr_png_path = Path(self.cxr_images[idx])
        mask_png_path = Path(self.mask_images[idx])
        img = np.array(Image.open(cxr_png_path).convert("RGB"))
        mask = np.array(Image.open(mask_png_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augs = self.transform(image=img, mask=mask)
            img = augs["image"]
            mask = augs["mask"]

        return img, mask