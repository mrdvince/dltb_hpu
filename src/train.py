import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from datamodules import ClassifierDataModule, UNETDataModule
from model import CLSModel, UNETModel


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        val_images, val_labels = val_batch
        # get predictions
        outputs = pl_module(val_images.cuda())
        preds = torch.argmax(outputs, dim=1)
        # predictions and labels

        df = pd.DataFrame(
            {"preds": preds.cpu().numpy(), "labels": val_labels.cpu().numpy()}
        )
        # incorrect predictions
        wrong_df = df[df.preds != df.labels]

        # log wrong predictions to wandb as table
        trainer.logger.experiment.log(
            {
                "wrong_preds": wandb.Table(
                    dataframe=wrong_df,
                    columns=["preds", "labels"],
                    allow_mixed_types=True,
                ),
                "global_step": trainer.global_step,
            },
        )


@hydra.main(config_path="configs", config_name="config")
def main(cfg):

    cls_data = ClassifierDataModule(config=cfg)
    cls_model = CLSModel(num_classes=cfg.model.num_classes)

    unet_data = UNETDataModule(config=cfg)
    unet_model = UNETModel(cfg=cfg)

    check_point = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="model_checkpoint_{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="valid/loss",
        mode="min",
        save_on_train_epoch_end=True,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="valid/loss", patience=10, verbose=True, mode="min"
    )
    wandb_logger = WandbLogger(
        project="dl4tb", save_dir=hydra.utils.get_original_cwd() + "/saved/"
    )
    if cfg.training.device == "tpu" and cfg.training.cores > 1:
        os.environ["WANDB_CONSOLE"] = "off"
    if cfg.training.model == "unet":
        unet_trainer = pl.Trainer(
            precision=16,
            amp_backend="native",
            logger=wandb_logger,
            callbacks=[
                check_point,
            ],
            default_root_dir=cfg.training.save_dir,
            tpu_cores=cfg.training.cores if cfg.training.device == "tpu" else None,
            gpus=1 if cfg.training.device == "gpu" else 0,
            fast_dev_run=False,
            limit_train_batches=cfg.training.limit_train_batches,
            limit_val_batches=cfg.training.limit_val_batches,
            max_epochs=cfg.training.max_epochs,
            log_every_n_steps=1,
            deterministic=cfg.training.deterministic,
        )
        unet_trainer.fit(unet_model, unet_data)

    else:
        cls_trainer = pl.Trainer(
            precision=16,
            amp_backend="native",
            logger=wandb_logger,
            callbacks=[
                check_point,
                early_stopping_callback,
                SamplesVisualisationLogger(cls_data),
            ],
            default_root_dir=cfg.training.save_dir,
            tpu_cores=cfg.training.cores if cfg.training.device == "tpu" else None,
            gpus=1 if cfg.training.device == "gpu" else 0,
            fast_dev_run=False,
            limit_train_batches=cfg.training.limit_train_batches,
            limit_val_batches=cfg.training.limit_val_batches,
            max_epochs=cfg.training.max_epochs,
            log_every_n_steps=cfg.training.log_every_n_steps,
            deterministic=cfg.training.deterministic,
        )
        cls_trainer.fit(cls_model, cls_data)


if __name__ == "__main__":
    seed = 69420
    pl.seed_everything(seed)
    main()
