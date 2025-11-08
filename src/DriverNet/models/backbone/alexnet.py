import torch.nn as nn
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet, AlexNet_Weights

def load_alexnet(
    num_classes: int = 10,
    pretrained: bool = True,
) -> AlexNet:
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    ctor = alexnet

    m = ctor(weights=weights)

    assert isinstance(m.fc, nn.Linear)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)

    return m
