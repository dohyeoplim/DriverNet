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
        f_global_avg,
        f_global_max,
        f_near_avg,
        f_near_max,
        f_mid_avg,
        f_mid_max,
        f_far_avg,
        f_far_max,
    ], dim=1)

class ChannelwiseRefineResidual(nn.Module):
    def __init__(self, feat_dim: int, p_drop2d: float = 0.1):
        super().__init__()
        self.dw = nn.Conv2d(
            feat_dim,
            feat_dim,
            kernel_size=3,
            padding=1,
            groups=feat_dim,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(feat_dim)
        self.act = nn.SiLU()
        self.drop2d = nn.Dropout2d(p_drop2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop2d(out)
        return x + out

class DepthGroupedHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.refine = ChannelwiseRefineResidual(feat_dim, p_drop2d=0.1)
        self.feat_dim = feat_dim

        global_dim = feat_dim * 2
        depth_dim = feat_dim * 6
        gate_hidden_dim = (global_dim + depth_dim) // 8

        self.gate_net = nn.Sequential(
            nn.Linear(global_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, depth_dim),
            nn.Sigmoid(),
        )

        interaction_dim = feat_dim * 2
        self.global_proj = nn.Linear(global_dim, interaction_dim)
        self.depth_proj = nn.Linear(depth_dim, interaction_dim)
        self.norm_global = nn.LayerNorm(interaction_dim)
        self.norm_depth = nn.LayerNorm(interaction_dim)

        hidden = interaction_dim // 2
        self.pre_norm = nn.LayerNorm(interaction_dim)
        self.dropout_pooled = nn.Dropout(dropout)
        self.fc1 = nn.Linear(interaction_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden, num_classes)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        feat = self.refine(feat)
        pooled = depth_group_pooled_features(feat, depth)

        global_features = pooled[:, : self.feat_dim * 2]
        depth_features = pooled[:, self.feat_dim * 2 :]

        gate = self.gate_net(global_features)
        gated_depth_features = depth_features * gate

        proj_global = self.norm_global(self.global_proj(global_features))
        proj_depth = self.norm_depth(self.depth_proj(gated_depth_features))
        interaction_vec = proj_global * proj_depth

        x = self.pre_norm(interaction_vec)
        x = self.dropout_pooled(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        return self.fc_out(x)
