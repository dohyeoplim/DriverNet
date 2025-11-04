from torchvision import transforms
from torchvision.transforms import InterpolationMode

class DriverTransforms:
    def __init__(self, img_size):
        self.img_size = img_size

    def get_transforms(self, train: bool = True):
        IMGNET_MEAN = [0.485, 0.456, 0.406]
        IMGNET_STD = [0.229, 0.224, 0.225]

        if train:
            train_tf = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(
                    brightness=(0.75, 1.25),
                    contrast=(0.75, 1.25),
                    saturation=(0.5, 1.5),
                    hue=(-0.05, 0.05)
                ),
                transforms.RandomAffine(
                    degrees=(-10, 10),
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=(-5, 5)
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.15,
                    p=0.5,
                    fill=0
                ),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.RandomSolarize(threshold=128, p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ])
            return train_tf

        else:
            val_tf = transforms.Compose([
                transforms.Resize(int(self.img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ])
            return val_tf
