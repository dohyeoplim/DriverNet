import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models import (
    vit_b_16, vit_b_32, vit_l_16, vit_l_32,
)
from torchvision.models.vision_transformer import (
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights
)
from typing import Literal
from src.DriverNet.models.heads import DepthGroupedViT

vit_model_names = Literal[
    "b_16", "b_32", "l_16", "l_32",
    "b_16_depthg", "b_32_depthg", "l_16_depthg", "l_32_depthg",
]

def load_vit(
    model_name: vit_model_names = "b_16",
    num_classes: int = 10,
    pretrained: bool = True,
    image_size: int = 224,
) -> nn.Module:
    depthg = model_name.endswith("_depthg")
    base_name = model_name.replace("_depthg", "")

    full_model_name = f"vit_{base_name}"

    weights_map = {
        "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,
        "vit_b_32": ViT_B_32_Weights.IMAGENET1K_V1,
        "vit_l_16": ViT_L_16_Weights.IMAGENET1K_V1,
        "vit_l_32": ViT_L_32_Weights.IMAGENET1K_V1,
    }
    weights = weights_map.get(full_model_name) if pretrained else None

    ctor_map = {
        "b_16": vit_b_16,
        "b_32": vit_b_32,
        "l_16": vit_l_16,
        "l_32": vit_l_32,
    }
    ctor = ctor_map[base_name]

    backbone = ctor(weights=weights, image_size=image_size)

    if depthg:
        patch_size = int(base_name.split('_')[-1])
        return DepthGroupedViT(
            backbone=backbone,
            num_classes=num_classes,
            patch_size=patch_size,
            image_size=image_size,
        )

    assert isinstance(backbone.heads.head, nn.Linear)
    in_features = backbone.heads.head.in_features
    backbone.heads.head = nn.Linear(in_features, num_classes)

    return backbone
