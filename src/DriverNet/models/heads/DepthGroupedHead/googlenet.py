import torch
from torch import nn
from torchvision.models import GoogLeNet
from src.DriverNet.models.heads import DepthGroupedHead
from src.DriverNet.utils.infer_feature_dim import infer_feat_dim

class DepthGroupedGoogLeNet(nn.Module):
    def __init__(self, backbone: GoogLeNet, num_classes: int, img_size: int = 224):
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.maxpool1,
            backbone.conv2,
            backbone.conv3,
            backbone.maxpool2,
            backbone.inception3a,
            backbone.inception3b,
            backbone.maxpool3,
            backbone.inception4a,
            backbone.inception4b,
            backbone.inception4c,
            backbone.inception4d,
            backbone.inception4e,
            backbone.maxpool4,
            backbone.inception5a,
            backbone.inception5b,
        )

        feat_dim = infer_feat_dim(self.features, img_size=img_size)

        self.head = DepthGroupedHead(feat_dim=feat_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.head(feat, depth)
        return logits
