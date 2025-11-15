import torch.nn as nn
import kornia.augmentation as K
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class Augmentations(nn.Module):
    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size
        self.augmentations = K.AugmentationSequential(
            K.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.0), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=(0.7, 1.3),
                contrast=(0.7, 1.3),
                saturation=(0.5, 1.5),
                hue=(-0.08, 0.08),
                p=0.8,
            ),
            K.RandomAffine(
                degrees=(-15, 15),
                translate=(0.1, 0.1),
                scale=(0.85, 1.15),
                shear=(-8, 8),
                p=0.8,
            ),
            K.RandomPerspective(distortion_scale=0.2, p=0.5),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.5), p=0.5),
            K.RandomSolarize(thresholds=0.5, p=0.1),
            K.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            data_keys=["input"],
        )

    def forward(self, x):
        return self.augmentations(x)

class DriverTransforms:
    def __init__(self, img_size):
        self.img_size = img_size

    def get_transforms(self, train: bool = True):
        if train:
            return transforms.Compose([
                transforms.Resize(self.img_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(self.img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
