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
    # f_global_max = F.adaptive_max_pool2d(feat, (1, 1)).squeeze()

    # f_near_avg = masked_avg_pool(feat, mask_near)
    f_near_max = masked_max_pool(feat, mask_near)

    # f_mid_avg = masked_avg_pool(feat, mask_mid)
    f_mid_max = masked_max_pool(feat, mask_mid)

    f_far_avg = masked_avg_pool(feat, mask_far)
    # f_far_max = masked_max_pool(feat, mask_far)

    return torch.cat([
        f_global_avg,
        f_near_max,
        f_mid_max,
        f_far_avg,
    ], dim=1)

def depth_group_maps(feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    depth_r, q1, q2 = compute_depth_groups(depth, feat)
    mask_near = depth_r <= q1
    mask_mid  = (depth_r > q1) & (depth_r <= q2)
    mask_far  = depth_r > q2

    near = mask_near.float()
    mid  = mask_mid.float()
    far  = mask_far.float()

    return torch.cat([near, mid, far], dim=1)

class DepthGroupedHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        dropout: float = 0.4,
    ):
        super().__init__()

        conv_out_dim = feat_dim * 4
        in_dim = conv_out_dim
        hidden1 = in_dim
        hidden2 = hidden1 // 2

        in_ch = feat_dim + 3
        mid_ch = feat_dim * 2

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.actc1 = nn.SiLU()

        self.conv2 = nn.Conv2d(mid_ch, conv_out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_out_dim)
        self.actc2 = nn.SiLU()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.pre_norm = nn.LayerNorm(in_dim)

        self.fc1 = nn.Linear(in_dim, hidden1)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden2, num_classes)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        depth_ch = depth_group_maps(feat, depth)
        x = torch.cat([feat, depth_ch], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actc1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actc2(x)

        x = self.global_pool(x).flatten(1)

        x = self.pre_norm(x)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        logits = self.fc_out(x)
        return logits

# class DepthGroupedHead(nn.Module):
#     def __init__(
#         self,
#         feat_dim: int,
#         num_classes: int,
#         dropout: float = 0.4,
#     ):
#         super().__init__()
#         in_dim = feat_dim * 4
#         hidden1 = in_dim
#         hidden2 = hidden1 // 2

#         self.pre_norm = nn.LayerNorm(in_dim)

#         self.fc1 = nn.Linear(in_dim, hidden1)
#         self.act1 = nn.SiLU()
#         self.dropout1 = nn.Dropout(dropout)

#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.act2 = nn.SiLU()
#         self.dropout2 = nn.Dropout(dropout)

#         self.fc_out = nn.Linear(hidden2, num_classes)

#     def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
#         pooled = depth_group_pooled_features(feat, depth)
#         x = self.pre_norm(pooled)

#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.dropout1(x)

#         x = self.fc2(x)
#         x = self.act2(x)
#         x = self.dropout2(x)

#         logits = self.fc_out(x)

#         return logits
