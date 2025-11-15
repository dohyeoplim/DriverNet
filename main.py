import argparse
from pathlib import Path
import torch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore:.*TF32.*"
warnings.filterwarnings("ignore", message=".*TF32.*")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriverNet")
    p.add_argument("--download-dataset", action="store_true", help="Download dataset from Kaggle")
    p.add_argument("--train", action="store_true", help="Train the model")
    p.add_argument("--test", action="store_true", help="Test the model")
    p.add_argument("--train-and-submit", action="store_true", help="Train the model and create a submission")
    p.add_argument("--checkpoint-path", type=str, default=None)
    return p.parse_args()

def main():

    args = parse_args()
    if args.download_dataset:
        from src.DriverNet.utils.download_dataset import download_kaggle_competition
        competition = "state-farm-distracted-driver-detection"
        out_dir = Path("./input")
        download_kaggle_competition(competition, out_dir=str(out_dir), unzip=True)
        return

    if args.train:
        from src.DriverNet.core.train import train
        train()
        return

    if args.test:
        from src.DriverNet.core.test import test
        test(checkpoint_path=args.checkpoint_path)
        return

    if args.train_and_submit:
        from src.DriverNet.core.train import train
        from src.DriverNet.core.test import test
        best_checkpoint_path = train()
        test(checkpoint_path=best_checkpoint_path)
        return

if __name__ == "__main__":
    main()
