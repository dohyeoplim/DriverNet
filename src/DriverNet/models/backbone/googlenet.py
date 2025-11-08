import torch.nn as nn
from torchvision.models import googlenet
from torchvision.models.googlenet import GoogLeNet, GoogLeNet_Weights

def load_googlenet(
    num_classes: int = 10,
    pretrained: bool = True,
) -> GoogLeNet:
    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    ctor = googlenet

    m = ctor(weights=weights)

    assert isinstance(m.fc, nn.Linear)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)

    return m
