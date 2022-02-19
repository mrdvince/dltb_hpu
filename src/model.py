import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

import wandb
from unet import UNET
from utils import get_device, mark_step, permute_params


class UNETModel(pl.LightningModule):
    def __init__(self, cfg):
        super(UNETModel, self).__init__()
        self.lr = cfg.model.lr
        self.save_hyperparameters()
        self.model = UNET(in_channels=3, out_channels=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.config = cfg
        self.dice_score = 0

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        logits = self(image)
        loss = self.criterion(logits, mask.unsqueeze(1))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask.unsqueeze(1)
        if self.config.training.device == "hpu":
            image, mask = image.to(torch.device("hpu"), non_blocking=False), mask.to(
                torch.device("hpu"), non_blocking=False
            )
        pred = self.forward(image)
        loss = self.criterion(pred, mask)
        preds = (torch.sigmoid(pred) > 0.5).float()
        dice_score = (2 * (preds * mask).sum()) / ((preds + mask).sum() + 1e-7)
        self.dice_score += dice_score
        mark_step(self.config.training.run_lazy_mode)
        self.log("valid/val_loss", loss, prog_bar=True, on_step=True)
        self.log("valid/dice_score", dice_score, prog_bar=True, on_step=True)

        if self.current_epoch > self.config.training.skip_first_n_eval:
            torchvision.utils.save_image(image, "images.png")
            torchvision.utils.save_image(preds, "pred.png")
            torchvision.utils.save_image(mask, "mask.png")

            wandb.log({"Images": wandb.Image("images.png")})
            wandb.log({"Predictions": wandb.Image("pred.png")})
            wandb.log({"Masks": wandb.Image("mask.png")})

        return {"mask": mask, "preds": preds}

    def configure_optimizers(self):
        self.model = self.model.to(get_device())
        permute_params(self.model, True, self.config.training.run_lazy_mode)
        return torch.optim.Adam(self.parameters(), lr=self.lr)
