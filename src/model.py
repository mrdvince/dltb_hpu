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


class CLSModel(pl.LightningModule):
    def __init__(self, num_classes, lr=0.001):
        super(CLSModel, self).__init__()
        self.lr = lr
        self.num_classes = num_classes

        self.save_hyperparameters()

        self.model = torchvision.models.resnet50(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        # metrics

        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.classification.f_beta.F1Score(
            num_classes=self.num_classes
        )
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_step=True)
        self.log("valid/precision_macro", precision_macro)
        self.log("valid/recall_macro", recall_macro)
        self.log("valid/precision_micro", precision_micro)
        self.log("valid/recall_micro", recall_micro)
        self.log("valid/f1", f1, prog_bar=True)
        return {"labels": labels, "logits": outputs}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])

        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
