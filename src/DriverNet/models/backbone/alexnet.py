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

    assert isinstance(m.classifier, nn.Sequential)
    last_layer = m.classifier[-1]
    assert isinstance(last_layer, nn.Linear)
    in_features = last_layer.in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)

    return m
