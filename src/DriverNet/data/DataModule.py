import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as L
from src.DriverNet.data.Transforms import DriverTransforms
from src.DriverNet.data.Dataset import DriverDataset

class DriverDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./input/imgs/train",
        csv_path: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 224,
        flip_p: float = 0.1,
        test_split: float = 0.15,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.flip_p = flip_p
        self.test_split = test_split

        self._tf = DriverTransforms(img_size=image_size)

        self.train_ds: Optional[DriverDataset] = None
        self.val_ds: Optional[DriverDataset] = None
        self.class_to_idx: Dict[str, int] = {}

    def prepare_data(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.data_dir}. Run `uv run main.py --download-dataset` first."
            )

    def setup(self, stage: Optional[str] = None):
        if self.csv_path and Path(self.csv_path).exists():
            df = pd.read_csv(self.csv_path)
        else:
            samples = []
            for class_dir in sorted(self.data_dir.glob("c*")):
                if class_dir.is_dir():
                    for img_path in class_dir.glob("*.jpg"):
                        samples.append({"classname": class_dir.name, "img": img_path.name})
            df = pd.DataFrame(samples)

        classes = sorted(df["classname"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        train_idx, val_idx = train_test_split(
            df.index,
            test_size=self.test_split,
            random_state=42,
            stratify=df["classname"],
        )
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)

        train_tf = self._tf.get_transforms(train=True)
        val_tf = self._tf.get_transforms(train=False)

        self.train_ds = DriverDataset(
            dataframe=train_df,
            root_dir=self.data_dir,
            class_to_idx=self.class_to_idx,
            transform=train_tf,
            flip_p=self.flip_p,
        )
        self.val_ds = DriverDataset(
            dataframe=val_df,
            root_dir=self.data_dir,
            class_to_idx=self.class_to_idx,
            transform=val_tf,
            flip_p=0.0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
