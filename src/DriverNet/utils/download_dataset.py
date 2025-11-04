import os
import sys
import zipfile
import subprocess
from pathlib import Path

def check_kaggle_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("Kaggle API credentials not found!")
        sys.exit(1)

    os.chmod(kaggle_json, 0o600)
    kaggle_dir.mkdir(parents=True, exist_ok=True)

def check_dataset_is_downloaded():
    pass

def download_kaggle_competition(dataset: str, out_dir: str = "./input", unzip: bool = True) -> None:
    check_kaggle_credentials()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset:", dataset)
    cmd = ["kaggle", "competitions", "download", "-c", dataset, "-p", str(out_dir_p)]
    subprocess.run(cmd, check=True)

    if unzip:
        for zf in out_dir_p.glob("*.zip"):
            print("Extracting", zf.name)
            with zipfile.ZipFile(zf, "r") as f:
                f.extractall(out_dir_p)
            zf.unlink(missing_ok=True)
        print("Done. Extracted to", out_dir_p)
