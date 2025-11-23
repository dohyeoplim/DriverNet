import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models.resnet import ResNet, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from typing import Literal
from src.DriverNet.models.heads import DepthGroupedResNet

resnet_model_names = Literal["50", "101", "152", "50_depthg", "101_depthg", "152_depthg"]

def load_resnet(
    model: str = "50",
    num_classes: int = 10,
    pretrained: bool = True,
) -> ResNet | DepthGroupedResNet:
    if "depthg" in model:
        base_name = model.replace("_depthg", "")
        weights = {
            "50": ResNet50_Weights.IMAGENET1K_V2,
            "101": ResNet101_Weights.IMAGENET1K_V2,
            "152": ResNet152_Weights.IMAGENET1K_V2,
        }[base_name] if pretrained else None

        ctor = {"50": resnet50, "101": resnet101, "152": resnet152}[base_name]
        backbone = ctor(weights=weights)

        depth_model = DepthGroupedResNet(
            backbone=backbone,
            num_classes=num_classes,
            model_name=base_name,
        )
        return depth_model

    weights = {
        "50": ResNet50_Weights.IMAGENET1K_V2,
        "101": ResNet101_Weights.IMAGENET1K_V2,
        "152": ResNet152_Weights.IMAGENET1K_V2,
    }[model] if pretrained else None

    ctor = {"50": resnet50, "101": resnet101, "152": resnet152}[model]
    m = ctor(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
