from pathlib import Path
from src.DriverNet.utils.download_dataset import download_kaggle_competition

def main():
    competition = "state-farm-distracted-driver-detection"
    out_dir = Path("./input")
    download_kaggle_competition(competition, out_dir=str(out_dir), unzip=True)
    return

if __name__ == "__main__":
    main()
