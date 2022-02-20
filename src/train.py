import hydra
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from datamodules import UNETDataModule
from model import UNETModel
from utils import PeriodicCheckpoint, load_hpu_library, set_env_params


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    # somehow wandb keeps yelling at me about this
    wandb.init(project="dltb_hpu", name=cfg.training.run_name)

    set_env_params(
        run_lazy_mode=cfg.training.run_lazy_mode, hpus_per_node=cfg.training.cores
    )
    load_hpu_library()

    unet_data = UNETDataModule(config=cfg)
    unet_model = UNETModel(cfg=cfg)
    model_ckpt = PeriodicCheckpoint(
        dirpath="checkpoints",
        filename="model_checkpoint_{epoch}",
        monitor="valid/dice_score",
        mode="max",
        save_top_k=3,
        save_weights_only=True,
        save_last=True,
        every_n=cfg.training.ckpt_every,
        pl_module=unet_model,
    )
    early_stopping = EarlyStopping(
        monitor="valid/dice_score",
        patience=cfg.training.patience,
        verbose=True,
        mode="max",
    )
    wandb_logger = WandbLogger(
        project="dltb_hpu", save_dir=hydra.utils.get_original_cwd() + "/saved/"
    )
    parallel_hpus = [torch.device("hpu")] * cfg.training.cores
    unet_trainer = pl.Trainer(
        hpus=cfg.training.cores if cfg.training.device else None,
        precision=16,
        logger=wandb_logger,
        callbacks=[model_ckpt, early_stopping],
        default_root_dir=cfg.training.save_dir,
        fast_dev_run=False,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=1,
        deterministic=cfg.training.deterministic,
        strategy=DDPPlugin(
            parallel_devices=parallel_hpus,
            bucket_cap_mb=cfg.training.bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if cfg.training.cores > 1
        else None,
    )
    # with torch.autograd.profiler.emit_nvtx():
    unet_trainer.fit(unet_model, unet_data)


if __name__ == "__main__":
    seed = 420
    pl.seed_everything(seed)
    main()
