import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import wandb

from unet import UNET


class UNETModel(pl.LightningModule):
    def __init__(self, cfg):
        super(UNETModel, self).__init__()
        self.lr = cfg.model.lr
        self.save_hyperparameters()
        self.model = UNET(in_channels=3, out_channels=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.dice_score = 0
        self.config = cfg

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        logits = self(image)
        loss = self.criterion(logits, mask.unsqueeze(1))
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
