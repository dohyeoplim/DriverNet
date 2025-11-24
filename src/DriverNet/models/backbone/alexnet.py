import torch.nn as nn
from torchvision.models import alexnet
from torchvision.models.alexnet import  AlexNet_Weights
from src.DriverNet.models.heads import DepthGroupedAlexNet
from typing import Literal

alexnet_model_names = Literal["", "_depthg"]

def load_alexnet(
    model: alexnet_model_names = "",
    num_classes: int = 10,
    pretrained: bool = True,
    image_size: int = 224,
):
    depthg = model.endswith("_depthg")

    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = alexnet(weights=weights)

    if depthg:
        return DepthGroupedAlexNet(backbone=backbone, num_classes=num_classes, img_size=image_size)

    assert isinstance(backbone.classifier, nn.Sequential)
    last = backbone.classifier[-1]
    assert isinstance(last, nn.Linear)
    backbone.classifier[-1] = nn.Linear(last.in_features, num_classes)

    return backbone
