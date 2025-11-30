import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
import torch.nn.functional as F

from src.DriverNet.models.base import BaseModel


def build_knn_and_search(
    all_features: np.ndarray, test_features: np.ndarray, max_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    print("Building KNN index...")
    knn_model = NearestNeighbors(n_neighbors=max_k, metric="cosine", n_jobs=-1)
    knn_model.fit(all_features)

    print("Searching neighbors...")
    max_distances, max_indices = knn_model.kneighbors(test_features)

    return max_distances, max_indices


def compute_and_ensemble_knn_probabilities(
    distances: np.ndarray,
    indices: np.ndarray,
    all_labels: np.ndarray,
    test_probs: np.ndarray,
    k: int,
    temperature: float = 0.1,
    alpha: float = 0.5,
) -> np.ndarray:
    weights = np.exp(-distances / temperature)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    num_test = len(distances)
    num_classes = test_probs.shape[1]
    knn_probs = np.zeros((num_test, num_classes))
    neighbor_labels = all_labels[indices]

    for j in range(k):
        lbls = neighbor_labels[:, j]
        w = weights[:, j]
        knn_probs[np.arange(num_test), lbls] += w

    knn_probs = knn_probs / (np.sum(knn_probs, axis=1, keepdims=True) + 1e-10)

    adjusted_probs = (1 - alpha) * test_probs + alpha * knn_probs
    adjusted_probs = np.clip(adjusted_probs, 1e-8, 1.0 - 1e-8)
    adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)

    return adjusted_probs


features_out = {}


def _feature_hook(module: nn.Module, input: Any, output: Any):
    features_out["features"] = output.detach().flatten(start_dim=1)


def _vit_feature_hook(module: nn.Module, input: Any, output: Any):
    features_out["features"] = output[:, 0].detach()


def get_feature_layer_name(model_name: str) -> str:
    model_name = model_name.lower()
    if "_depthg" in model_name:
        return "model.features"
    if "resnet" in model_name or "vgg" in model_name or "alexnet" in model_name or "googlenet" in model_name:
        return "model.avgpool"
    if "vit" in model_name:
        return "model.encoder.ln"
    raise ValueError(f"Feature layer not defined for model {model_name}")


def extract_features(
    model: BaseModel, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    feature_layer_name = get_feature_layer_name(model.hparams.name) # type: ignore

    module = model
    try:
        for part in feature_layer_name.split("."):
            module = getattr(module, part)
    except AttributeError:
        raise AttributeError(f"Could not find layer {feature_layer_name} in model.")

    hook_fn = _vit_feature_hook if "vit" in model.hparams.name.lower() else _feature_hook # type: ignore
    handle = module.register_forward_hook(hook_fn)

    all_features = []
    all_labels = []
    all_probs = []
    all_img_names = []
    is_predict = False

    for batch in tqdm(dataloader, desc="Extracting features"):
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)

        pixel_values = batch["pixel_values"]
        depth = batch.get("depth")

        with torch.no_grad():
            if model._uses_depth: # type: ignore
                teacher_logits = model.teacher_forward(pixel_values, depth)
            else:
                teacher_logits = model.teacher_forward(pixel_values)

        all_features.append(features_out["features"].cpu())

        if "labels" in batch:
            all_labels.append(batch["labels"].cpu())
        else:
            is_predict = True

        if "img_name" in batch:
            all_img_names.extend(batch["img_name"])

        all_probs.append(torch.softmax(teacher_logits, dim=1).cpu())

    handle.remove()

    res = {
        "features": torch.cat(all_features).numpy(),
        "probs": torch.cat(all_probs).numpy(),
    }
    if not is_predict:
        res["labels"] = torch.cat(all_labels).numpy()
    if is_predict:
        res["img_names"] = all_img_names

    return res
