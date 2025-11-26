import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.vgg import VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights
from src.DriverNet.models.heads import DepthGroupedVGG
from typing import Literal

vgg_model_names = Literal["16", "16_bn", "19", "19_bn", "16_depthg", "16_bn_depthg", "19_depthg", "19_bn_depthg"]

def load_vgg(
    model: vgg_model_names = "16",
    num_classes: int = 10,
    pretrained: bool = True,
    image_size: int = 224,
):
    depthg = model.endswith("_depthg")
    base_name = model.replace("_depthg", "")

    weights_map = {
        "16": VGG16_Weights.IMAGENET1K_V1,
        "16_bn": VGG16_BN_Weights.IMAGENET1K_V1,
        "19": VGG19_Weights.IMAGENET1K_V1,
        "19_bn": VGG19_BN_Weights.IMAGENET1K_V1,
    }
    weights = weights_map[base_name] if pretrained else None

    ctor = {
        "16": vgg16,
        "16_bn": vgg16_bn,
        "19": vgg19,
        "19_bn": vgg19_bn,
    }[base_name]

    backbone = ctor(weights=weights)

    if depthg:
        return DepthGroupedVGG(backbone, num_classes=num_classes, img_size=image_size)

    assert isinstance(backbone.classifier, nn.Sequential)
    last_layer = backbone.classifier[-1]
    assert isinstance(last_layer, nn.Linear)

    in_features = last_layer.in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    return backbone
