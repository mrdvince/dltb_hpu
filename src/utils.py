import os
import shutil
from copy import deepcopy
from glob import glob
from pathlib import Path

import opendatasets as od
import pytorch_lightning as pl
import torch
from PIL import Image, ImageChops
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm


def download(dataset_url, data_dir):
    od.download(dataset_url, data_dir)


def copy_cxr_merge_masks(raw_image_dir, cxr_dir, mask_dir):
    image_paths = glob(os.path.join(raw_image_dir, "*.png"))
    # fmt: off
    images_with_masks_paths = [
        (image_path,os.path.join("/".join(image_path.split("/")[:-2]),"ManualMask","leftMask", os.path.basename(image_path)),
         os.path.join("/".join(image_path.split("/")[:-2]),"ManualMask","rightMask",os.path.basename(image_path))) for image_path in image_paths
        ]
    # fmt: on
    mask_path = Path(mask_dir)
    mask_path.mkdir(exist_ok=True, parents=True)

    cxr_path = Path(cxr_dir)
    cxr_path.mkdir(exist_ok=True, parents=True)

    for cxr, left, right in tqdm(images_with_masks_paths):
        left = Image.open(left).convert("L")
        right = Image.open(right).convert("L")
        seg_img = ImageChops.add(left, right)
        filename = Path(cxr).name
        shutil.copy(cxr, cxr_path / filename)
        seg_img.save(mask_path / filename)


def get_data(
    cxr_dir="data/proc_seg/cxr_pngs",
    mask_dir="data/proc_seg/mask_pngs",
    data_dir="data",
    raw_image_dir="data/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png",
):
    download(
        "https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities",
        data_dir=data_dir,
    )
    copy_cxr_merge_masks(
        raw_image_dir=raw_image_dir,
        cxr_dir=cxr_dir,
        mask_dir=mask_dir,
    )


# https://github.com/HabanaAI/Model-References/blob/master/PyTorch/computer_vision/segmentation/Unet/utils/utils.py
def set_env_params(run_lazy_mode, hpus_per_node=1):
    os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    os.environ["PT_HPU_ENABLE_SYNC_OUTPUT_HOST"] = "false"
    if run_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    if hpus_per_node > 1:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "hccl"


def load_hpu_library():
    from habana_frameworks.torch.utils.library_loader import load_habana_module

    load_habana_module()


def get_device():
    return torch.device("hpu")


def mark_step(is_lazy_mode):
    if is_lazy_mode:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()


def permute_4d_5d_tensor(tensor, to_filters_last):
    import habana_frameworks.torch.core as htcore

    if htcore.is_enabled_weight_permute_pass() is True:
        return tensor
    if tensor.ndim == 4:
        if to_filters_last:
            tensor = tensor.permute((2, 3, 1, 0))
        else:
            tensor = tensor.permute((3, 2, 0, 1))  # permute RSCK to KCRS
    elif tensor.ndim == 5:
        if to_filters_last:
            tensor = tensor.permute((2, 3, 4, 1, 0))
        else:
            tensor = tensor.permute((4, 3, 0, 1, 2))  # permute RSTCK to KCRST
    return tensor


def change_state_dict_device(state_dict, to_device):
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            state_dict[name] = param.to(to_device)
    return state_dict


def adjust_tensors_for_save(
    state_dict, optimizer_states, to_device, to_filters_last, lazy_mode, permute
):
    if permute:
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param.data = permute_4d_5d_tensor(param.data, to_filters_last)
        mark_step(lazy_mode)

    change_state_dict_device(state_dict, to_device)

    for state in optimizer_states.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(to_device)


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        filepath=None,
        monitor=None,
        verbose=False,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="auto",
        period=1,
        prefix="",
        dirpath=None,
        filename=None,
        every_n=10,
        first_n=10,
        pl_module=None,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            every_n_epochs=period,
        )
        self.every_n = every_n
        self.first_n = first_n
        self.pl_module = pl_module

    def restore_tensors_for_ckpt(self, pl_module, state_dict):
        assert pl_module.config.training.device == "hpu"

        pl_module.model.load_state_dict(state_dict)
        adjust_tensors_for_save(
            pl_module.model.state_dict(),
            pl_module.optimizers().state,
            to_device="hpu",
            to_filters_last=True,
            lazy_mode=pl_module.config.training.run_lazy_mode,
            permute=False,
        )

    def _save_model(self, trainer: "pl.Trainer", filepath: str) -> None:
        # make paths
        if trainer.should_rank_save_checkpoint:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the trainer
        trainer.save_checkpoint(filepath, self.save_weights_only)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Save a copy of state_dict and restore after save is finished
        if pl_module.config.training.device == "hpu":
            state_dict = deepcopy(
                change_state_dict_device(pl_module.model.state_dict(), "cpu")
            )

        super(PeriodicCheckpoint, self).on_validation_end(trainer, pl_module)
        # save backups
        if self.every_n:
            epoch = trainer.current_epoch
            if epoch < self.first_n or epoch % self.every_n == 0:
                filepath = os.path.join(self.dirpath, f"backup_epoch_{epoch}.pt")
                # print("save backup chekpoint: ", filepath)
                # self._save_model(filepath, trainer, pl_module)
                # pl 1.4
                self._save_model(trainer, filepath)
        if pl_module.config.training.device == "hpu":
            self.restore_tensors_for_ckpt(pl_module, state_dict)

    def save_checkpoint(self, trainer: "pl.Trainer"):
        pl_module = self.pl_module
        if pl_module.config.training.device == "hpu":
            state_dict = deepcopy(
                change_state_dict_device(pl_module.model.state_dict(), "cpu")
            )
        super(PeriodicCheckpoint, self).save_checkpoint(trainer)
        if pl_module.config.training.device == "hpu":
            self.restore_tensors_for_ckpt(pl_module, state_dict)


def permute_params(model, to_filters_last, lazy_mode):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = permute_4d_5d_tensor(param.data, to_filters_last)
    mark_step(lazy_mode)


if __name__ == "__main__":
    get_data()
