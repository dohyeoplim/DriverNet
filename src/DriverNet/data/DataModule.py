import lightning as L
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.DriverNet.data.Dataset import DriverDataset
from src.DriverNet.data.Transforms import DriverTransforms

class DriverDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        prefetch_factor: int,
        image_size: int,
        flip_p: float,
        validation_split: Optional[float] = None,
        original_data_dir: Optional[str] = None,
        processed_data_dir: Optional[str] = None,
        processed_hard_data_dir: Optional[str] = None,
        processed_depth_data_dir: Optional[str] = None,
        csv_path: Optional[str] = None,
        predict_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.original_data_dir = Path(original_data_dir) if original_data_dir else None
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else None
        self.processed_hard_data_dir = Path(processed_hard_data_dir) if processed_hard_data_dir else None
        self.processed_depth_data_dir = Path(processed_depth_data_dir) if processed_depth_data_dir else None
        self.predict_dir = Path(predict_dir) if predict_dir else None
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor if prefetch_factor else None
        self.image_size = image_size
        self.flip_p = flip_p
        self.validation_split = validation_split

        self._tf = DriverTransforms(img_size=image_size)

        self.train_ds: Optional[DriverDataset] = None
        self.val_ds: Optional[DriverDataset] = None
        self.predict_ds: Optional[DriverDataset] = None
        self.class_to_idx: Dict[str, int] = {}

    def prepare_data(self):
        if not any([self.original_data_dir, self.processed_data_dir, self.processed_hard_data_dir, self.processed_depth_data_dir, self.predict_dir]):
            raise ValueError("At least one data directory must be provided.")

    def setup(self, stage: Optional[str] = None):
        if stage == "predict":
            if self.predict_dir is None:
                raise ValueError("predict_dir must be set for predict stage.")
            if not self.predict_dir.exists():
                raise FileNotFoundError(f"Test dataset folder not found: {self.predict_dir}")

            test_images = sorted(list(self.predict_dir.glob("*.jpg")))
            predict_df = pd.DataFrame({"img": [p.name for p in test_images]})
            val_tf = self._tf.get_transforms(train=False)

            self.predict_ds = DriverDataset(
                dataframe=predict_df,
                original_root_dir=self.predict_dir,
                transform=val_tf,
                is_predict=True,
            )
            return

        if self.train_ds is None or self.val_ds is None:
            assert self.original_data_dir is not None

            if not self.csv_path or not Path(self.csv_path).exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            df = pd.read_csv(self.csv_path)
            classes = sorted(df["classname"].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

            if self.validation_split is None or not (0.0 < self.validation_split < 0.5):
                raise ValueError("validation_split must be set to a value in (0, 0.5).")

            n_splits = int(round(1.0 / self.validation_split))
            n_splits = max(2, n_splits)

            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            X = df.index.values
            y = df["classname"].values
            groups = df["subject"].values
            train_idx, val_idx = next(sgkf.split(X, y, groups))

            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_tf = self._tf.get_transforms(train=True)
            val_tf = self._tf.get_transforms(train=False)

            self.train_ds = DriverDataset(
                dataframe=train_df.reset_index(drop=True),
                original_root_dir=self.original_data_dir,
                processed_root_dir=self.processed_data_dir,
                processed_hard_root_dir=self.processed_hard_data_dir,
                class_to_idx=self.class_to_idx,
                transform=train_tf,
                processed_transform=val_tf,
                flip_p=self.flip_p,
            )

            self.val_ds = DriverDataset(
                dataframe=val_df.reset_index(drop=True),
                original_root_dir=self.original_data_dir,
                class_to_idx=self.class_to_idx,
                transform=val_tf,
                is_val=True,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.predict_ds is not None
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def get_test_image_names(self) -> list[str]:
        assert self.predict_ds is not None, "Call setup(stage='predict') first."
        return list(self.predict_ds.df["img"])  # type: ignore[attr-defined]
