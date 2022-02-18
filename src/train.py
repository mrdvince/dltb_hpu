import os

import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datamodules import UNETDataModule
from model import UNETModel


@hydra.main(config_path="configs", config_name="config")
def main(cfg):

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


if __name__ == "__main__":
    seed = 69420
    pl.seed_everything(seed)
    main()
