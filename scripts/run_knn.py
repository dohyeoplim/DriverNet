import argparse
import os
import torch
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import numpy as np

from src.DriverNet.models.base import BaseModel
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.utils.knn import extract_features, build_knn_and_search, compute_and_ensemble_knn_probabilities
from src.DriverNet.utils.submission import create_submission

def run_knn(checkpoint_path: str, output_dir: str, k_values: list[int], alpha: float, temp: float):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}...")
    model = BaseModel.load_from_checkpoint(checkpoint_path=checkpoint_path, **cfg.model)

    print("Setting up datamodules...")
    dm_train = DriverDataModule(**cfg.data)
    dm_train.setup()

    train_loader = dm_train.train_dataloader()
    val_loader = dm_train.val_dataloader()

    print("Extracting features from training data...")
    train_features_data = extract_features(model, train_loader, device)
    print("Extracting features from validation data...")
    val_features_data = extract_features(model, val_loader, device)

    all_features = np.concatenate([train_features_data["features"], val_features_data["features"]])
    all_labels = np.concatenate([train_features_data["labels"], val_features_data["labels"]])

    print(f"Total features for KNN library: {all_features.shape[0]}")

    dm_pred = DriverDataModule(**cfg.data_test)
    dm_pred.setup(stage="predict")
    predict_loader = dm_pred.predict_dataloader()

    print("Extracting features from test data...")
    test_data = extract_features(model, predict_loader, device)
    test_features = test_data["features"]
    test_probs = test_data["probs"]
    test_img_names = test_data["img_names"]

    print("Starting KNN post-processing...")
    max_k = max(k_values)
    distances, indices = build_knn_and_search(all_features, test_features, max_k)

    os.makedirs(output_dir, exist_ok=True)

    for k in k_values:
        print(f"Processing for k={k}...")

        k_distances = distances[:, :k]
        k_indices = indices[:, :k]

        adjusted_probs = compute_and_ensemble_knn_probabilities(
            distances=k_distances,
            indices=k_indices,
            all_labels=all_labels,
            test_probs=test_probs,
            k=k,
            temperature=temp,
            alpha=alpha,
        )

        create_submission(submission_outputs, path=submission_path)

    print("KNN post-processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KNN post-processing on a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--outdir", type=str, default="./output/knn_submissions", help="Directory to save submission files.")
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 200], help="List of k values for KNN.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for ensembling KNN probs with model probs.")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for KNN probability calculation.")

    args = parser.parse_args()

    run_knn(
        checkpoint_path=args.checkpoint,
        output_dir=args.outdir,
        k_values=args.k,
        alpha=args.alpha,
        temp=args.temp,
    )
