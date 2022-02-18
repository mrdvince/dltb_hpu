import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
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
        logits = self.model(image)
        loss = self.criterion(logits, mask.unsqueeze(1))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask.unsqueeze(1)
        logits = self.model(image)
        loss = self.criterion(logits, mask)
        preds = (torch.sigmoid(logits) > 0.5).float()
        dice_score = (2 * (preds * mask).sum()) / ((preds + mask).sum() + 1e-7)
        self.dice_score += dice_score
        torchvision.utils.save_image(image, "images.png")
        torchvision.utils.save_image(preds, "pred.png")
        torchvision.utils.save_image(mask, "mask.png")

        wandb.log({"Images": wandb.Image("images.png")})
        wandb.log({"Predictions": wandb.Image("pred.png")})
        wandb.log({"Masks": wandb.Image("mask.png")})

        self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/dice_score", dice_score, prog_bar=True, on_step=True)
        return {"mask": mask, "preds": preds}

    def validation_epoch_end(self, outputs):

        preds = torch.cat([x["preds"] for x in outputs])
        mask = torch.cat([x["mask"] for x in outputs])

        torchvision.utils.save_image(preds, "epoch_preds.png")
        torchvision.utils.save_image(mask, "epoch_mask.png")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
