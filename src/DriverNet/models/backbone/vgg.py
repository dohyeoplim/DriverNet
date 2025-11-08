import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.vgg import VGG, VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights
from typing import Literal

vgg_model_names = Literal["16", "16_bn", "19", "19_bn"]

def load_vgg(
    model: vgg_model_names = "16",
    num_classes: int = 10,
    pretrained: bool = True,
) -> VGG:
    weights = {
        "16": VGG16_Weights.IMAGENET1K_V1 ,
        "16_bn": VGG16_BN_Weights.IMAGENET1K_V1,
        "19": VGG19_Weights.IMAGENET1K_V1,
        "19_bn": VGG19_BN_Weights.IMAGENET1K_V1
    }[model] if pretrained else None

    ctor = {
        "16": vgg16,
        "16_bn": vgg16_bn,
        "19": vgg19,
        "19_bn": vgg19_bn
    }[model]

    m = ctor(weights=weights)

    assert isinstance(m.fc, nn.Linear)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)

    return m
