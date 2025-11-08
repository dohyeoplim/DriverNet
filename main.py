import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriverNet")
    p.add_argument("--download-dataset", action="store_true", help="Download dataset from Kaggle")
    p.add_argument("--train", choices=["teacher", "student"], help="Train [teacher/student] model")
    p.add_argument("--test", choices=["teacher", "student"], help="Test [teacher/student] model")
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
        train(args.train)
        return

    if args.test:
        from src.DriverNet.core.test import test
        test(args.test)
        return

if __name__ == "__main__":
    main()
