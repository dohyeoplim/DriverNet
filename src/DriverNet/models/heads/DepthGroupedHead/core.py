import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DriverNet.utils.depth_utils import compute_depth_groups, masked_avg_pool

def depth_group_pooled_features(feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    B, C, Hf, Wf = feat.shape

    depth_r, q1, q2 = compute_depth_groups(depth, feat)
    mask_near = depth_r <= q1
    mask_mid  = (depth_r > q1) & (depth_r <= q2)
    mask_far  = depth_r > q2

    f_global = feat.mean(dim=(2, 3))

    f_near = masked_avg_pool(feat, mask_near)
    f_mid  = masked_avg_pool(feat, mask_mid)
    f_far  = masked_avg_pool(feat, mask_far)

    return torch.cat([f_global, f_near, f_mid, f_far], dim=1)

# class DepthGroupedHead(nn.Module):
#     def __init__(self, feat_dim: int, num_classes: int):
#         super().__init__()
#         in_dim = feat_dim * 4
#         hidden = in_dim

#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.BatchNorm1d(hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(hidden, num_classes),
#         )

#     def forward(self, feat: torch.Tensor, depth: torch.Tensor):
#         pooled = depth_group_pooled_features(feat, depth)
#         return self.mlp(pooled)

class DepthGroupedHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        in_dim = feat_dim * 4
        hidden = in_dim // 2

        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)

        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)

        self.fc_out = nn.Linear(hidden, num_classes)

    def forward(self, feat, depth):
        pooled = depth_group_pooled_features(feat, depth)

        x = self.fc1(pooled)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.dropout2(x)

        return self.fc_out(x)
