import os
import shutil
from glob import glob
from pathlib import Path

import opendatasets as od
from PIL import Image, ImageChops
from tqdm.auto import tqdm


# https://github.com/HabanaAI/Model-References/blob/master/PyTorch/computer_vision/segmentation/Unet/utils/utils.py
def set_env_params(run_lazy_mode, hpus_per_node=1):
    os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    os.environ["PT_HPU_ENABLE_SYNC_OUTPUT_HOST"] = "false"

    os.environ["HCL_CPU_AFFINITY"] = "1"

    if run_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    if hpus_per_node > 1:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "hccl"


def load_hpu_library():
    from habana_frameworks.torch.utils.library_loader import load_habana_module

    load_habana_module()


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
