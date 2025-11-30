import torch
import torch.nn as nn
import kornia.augmentation as K
from typing import Optional, Tuple

class Augmentations(nn.Module):
    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size

        self.geometric_augs = K.AugmentationSequential(
            K.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0), p=1.0, same_on_batch=False, align_corners=True),
            K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
            K.RandomAffine(
                degrees=(-20, 20),
                translate=(0.15, 0.15),
                scale=(0.8, 1.2),
                shear=(-10, 10),
                p=0.8,
                same_on_batch=False,
                align_corners=True,
            ),
            K.RandomPerspective(distortion_scale=0.25, p=0.5, same_on_batch=False, align_corners=True),
            K.RandomErasing(p=0.25, scale=(0.02, 0.25), same_on_batch=False),
            data_keys=["input", "mask"],
        )

        self.photometric_augs = K.AugmentationSequential(
            K.ColorJitter(
                brightness=(0.7, 1.3),
                contrast=(0.7, 1.3),
                saturation=(0.5, 1.5),
                hue=(-0.08, 0.08),
                p=0.8,
            ),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.5), p=0.5),
            K.RandomSolarize(thresholds=0.5, p=0.1),
            data_keys=["input"],
        )

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, depth: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if depth is not None:
            x, depth = self.geometric_augs(x, depth)
        else:
            x = self.geometric_augs(x, None)

        x = self.photometric_augs(x)

        return x, depth
