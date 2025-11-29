import torch
import torch.nn.functional as F

def masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().expand_as(feat)
    num = (feat * mask).sum(dim=(2, 3))
    den = mask.sum(dim=(2, 3))
    return num / (den + 1e-6)

def masked_max_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(feat)
    feat_masked = torch.where(mask, feat, torch.tensor(-torch.inf, dtype=feat.dtype, device=feat.device))
    return F.adaptive_max_pool2d(feat_masked, (1, 1)).squeeze()

def compute_depth_groups(depth: torch.Tensor, feat: torch.Tensor):
    B, C, Hf, Wf = feat.shape
    depth_r = F.interpolate(depth, (Hf, Wf), mode="bicubic", align_corners=False)
    depth_flat = depth_r.view(B, -1)

    qs = torch.quantile(depth_flat, torch.tensor([0.33, 0.66], device=depth.device), dim=1)
    q1 = qs[0].view(B, 1, 1, 1)
    q2 = qs[1].view(B, 1, 1, 1)
    return depth_r, q1, q2
