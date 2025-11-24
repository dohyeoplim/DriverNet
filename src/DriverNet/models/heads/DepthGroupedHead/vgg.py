from torch import nn
from torchvision.models import VGG
from src.DriverNet.models.heads import DepthGroupedHead
from src.DriverNet.utils.infer_feature_dim import infer_feat_dim

class DepthGroupedVGG(nn.Module):
    def __init__(self, backbone: VGG, num_classes: int, img_size: int = 224):
        super().__init__()

        self.features = backbone.features

        feat_dim = infer_feat_dim(self.features, img_size=img_size)

        self.head = DepthGroupedHead(feat_dim=feat_dim, num_classes=num_classes)

    def forward(self, x, depth):
        feat = self.features(x)
        logits = self.head(feat, depth)
        return logits
