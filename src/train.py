import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datamodules import UNETDataModule
from model import UNETModel
from utils import load_hpu_library, set_env_params


@hydra.main(config_path="configs", config_name="config")
def main(cfg):

    set_env_params(run_lazy_mode=False, hpus_per_node=cfg.training.cores)
    load_hpu_library()

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
    wandb_logger = WandbLogger(project="hpu" + "/saved/")
    unet_trainer = pl.Trainer(
        hpus=cfg.training.cores if cfg.training.device else None,
        precision=16,
        logger=wandb_logger,
        callbacks=[
            check_point,
        ],
        default_root_dir=cfg.training.save_dir,
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
