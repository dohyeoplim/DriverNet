import torch
import torch.nn.functional as F

def masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    mask_sum = mask.sum(dim=(-1, -2), keepdim=True)
    mask_safe = mask.float().expand_as(feat)

    num = (feat * mask_safe).sum(dim=(2, 3))
    den = mask_safe.sum(dim=(2, 3))

    safe_avg = torch.where(
        (den > 0),
        num / (den + 1e-6),
        torch.zeros_like(num)
    )
    return safe_avg

def masked_max_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    mask_sum = mask.sum(dim=(-1, -2), keepdim=True)

    feat_masked = torch.where(
        mask.expand_as(feat),
        feat,
        torch.full_like(feat, -torch.inf)
    )

    pooled = F.adaptive_max_pool2d(feat_masked, (1, 1)).squeeze(-1).squeeze(-1)

    condition = mask_sum.squeeze(-1).squeeze(-1) > 0

    safe_max = torch.where(
        condition,
        pooled,
        torch.zeros_like(pooled)
    )
    return safe_max

def compute_depth_groups(depth: torch.Tensor, feat: torch.Tensor):
    B, C, Hf, Wf = feat.shape
    depth_r = F.interpolate(depth, (Hf, Wf), mode="bicubic", align_corners=True)
    depth_flat = depth_r.view(B, -1)

    qs = torch.quantile(depth_flat, torch.tensor([0.33, 0.66], device=depth.device), dim=1)
    q1 = qs[0].view(B, 1, 1, 1)
    q2 = qs[1].view(B, 1, 1, 1)
    return depth_r, q1, q2
