import torch
import torch.nn as nn

def infer_feat_dim(features: nn.Module, img_size: int = 224) -> int:
    was_training = features.training
    device = next(features.parameters()).device

    features.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, device=device)
        feat = features(x)

    if was_training:
        features.train()

    if feat.ndim != 4:
        raise RuntimeError(f"Expected 4D feature map, got shape {feat.shape}")

    return int(feat.shape[1])
