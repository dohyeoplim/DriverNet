import torch.nn as nn
from torchvision.models import googlenet
from torchvision.models.googlenet import GoogLeNet, GoogLeNet_Weights
from src.DriverNet.models.heads import DepthGroupedGoogLeNet
from typing import Literal

googlenet_model_names = Literal["", "_depthg"]

def load_googlenet(
    model: googlenet_model_names = "",
    num_classes: int = 10,
    pretrained: bool = True,
    image_size: int = 224,
) -> nn.Module:
    depthg = model.endswith("_depthg")

    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = googlenet(weights=weights)

    if depthg:
        return DepthGroupedGoogLeNet(backbone=backbone, num_classes=num_classes, img_size=image_size)

    assert isinstance(backbone.fc, nn.Linear)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    return backbone
