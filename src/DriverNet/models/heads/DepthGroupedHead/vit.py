import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer
from src.DriverNet.models.heads import DepthGroupedHead
import math

class DepthGroupedViT(nn.Module):
    def __init__(self, backbone: VisionTransformer, num_classes: int, patch_size: int, image_size: int = 224):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.image_size = image_size

        feat_dim = backbone.hidden_dim
        self.head = DepthGroupedHead(feat_dim=feat_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x = self.backbone._process_input(x)
        n = x.shape[0]

        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        x = x[:, 1:]

        h = w = self.image_size // self.patch_size
        c = self.backbone.hidden_dim

        feat = x.permute(0, 2, 1).reshape(n, c, h, w)

        logits = self.head(feat, depth)
        return logits
