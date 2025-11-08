import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models.resnet import ResNet, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from typing import Literal

resnet_model_names = Literal["50", "101", "152"]

def load_resnet(
    model: str = "50",
    num_classes: int = 10,
    pretrained: bool = True,
) -> ResNet:
    weights = {
        "50": ResNet50_Weights.IMAGENET1K_V2,
        "101": ResNet101_Weights.IMAGENET1K_V2,
        "152": ResNet152_Weights.IMAGENET1K_V2,
    }[model] if pretrained else None

    ctor = {
        "50": resnet50,
        "101": resnet101,
        "152": resnet152,
    }[model]

    m = ctor(weights=weights)

    assert isinstance(m.fc, nn.Linear)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)

    return m
