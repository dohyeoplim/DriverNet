import torch
from torch import nn
from torchvision.models import ResNet
from src.DriverNet.models.heads import DepthGroupedHead

class DepthGroupedResNet(nn.Module):
    def __init__(self, backbone: ResNet, num_classes: int, model_name: str):
        super().__init__()
        self.model_name = model_name

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        c = 1024

        self.head = DepthGroupedHead(feat_dim=c, num_classes=num_classes)

    def forward(self, x: torch.Tensor, depth: torch.Tensor):
        feat = self.stem(x)
        logits = self.head(feat, depth)
        return logits
