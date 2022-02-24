import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from .unet import UNET


def load_model(model_path):
    model = UNET(in_channels=3, out_channels=1)

    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    return model


def predict(model, img, filename, static_dir):
    tsfm = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = tsfm(img)
    img_tensor = img_tensor.unsqueeze(0)
    preds = (torch.sigmoid(model(img_tensor)) > 0.5).float()
    img = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
    mask = transforms.ToPILImage()(preds.squeeze(0).cpu())
    mask.save(os.path.join(static_dir, "mask_" + filename))
    img.save(os.path.join(static_dir, "img_" + filename))
    cv2.imwrite(
        os.path.join(static_dir, "weighted_" + filename),
        cv2.addWeighted(np.array(img)[..., 2], 1.0, np.array(mask), 0.7, 1),
    )
    return filename
