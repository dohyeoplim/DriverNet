import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models import (
    vit_b_16, vit_b_16, vit_b_32, vit_l_16, vit_l_32,
)
from torchvision.models.vision_transformer import (
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights
)
from typing import Literal

vit_model_names = Literal["vit_b_16","vit_b_32","vit_l_16","vit_l_32"]

def load_vit(
    model_name: vit_model_names = "vit_b_16",
    num_classes: int = 10,
    pretrained: bool = True,
) -> VisionTransformer:
    weights = {
        "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,
        "vit_b_32": ViT_B_32_Weights.IMAGENET1K_V1,
        "vit_l_16": ViT_L_16_Weights.IMAGENET1K_V1,
        "vit_l_32": ViT_L_32_Weights.IMAGENET1K_V1,
    }[model_name] if pretrained else None

    ctor = {
        "vit_b_16": vit_b_16,
        "vit_b_32": vit_b_32,
        "vit_l_16": vit_l_16,
        "vit_l_32": vit_l_32,
    }[model_name]

    model = ctor(weights=weights)

    assert isinstance(model.heads.head, nn.Linear)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model
