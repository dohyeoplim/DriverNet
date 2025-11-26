import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DriverNet.utils.depth_utils import compute_depth_groups, masked_avg_pool, masked_max_pool

def depth_group_pooled_features(feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    depth_r, q1, q2 = compute_depth_groups(depth, feat)
    mask_near = depth_r <= q1
    mask_mid  = (depth_r > q1) & (depth_r <= q2)
    mask_far  = depth_r > q2

    f_global_avg = feat.mean(dim=(2, 3))
    f_global_max = F.adaptive_max_pool2d(feat, (1, 1)).squeeze()

    f_near_avg = masked_avg_pool(feat, mask_near)
    f_near_max = masked_max_pool(feat, mask_near)

    f_mid_avg = masked_avg_pool(feat, mask_mid)
    f_mid_max = masked_max_pool(feat, mask_mid)

    f_far_avg = masked_avg_pool(feat, mask_far)
    f_far_max = masked_max_pool(feat, mask_far)

    return torch.cat([
        f_global_avg, f_global_max,
        f_near_avg, f_near_max,
        f_mid_avg, f_mid_max,
        f_far_avg, f_far_max,
    ], dim=1)

class DepthGroupedHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_dim = feat_dim * 8
        hidden1 = in_dim // 2
        hidden2 = hidden1 // 2

        self.pre_norm = nn.LayerNorm(in_dim)

        self.fc1 = nn.Linear(in_dim, hidden1)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden2, num_classes)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        pooled = depth_group_pooled_features(feat, depth)
        x = self.pre_norm(pooled)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        logits = self.fc_out(x)

        return logits
