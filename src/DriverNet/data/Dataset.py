import random
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FLIP_REMAP = {1: 3, 3: 1, 2: 4, 4: 2}

class DriverDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        original_root_dir: Path,
        processed_root_dir: Optional[Path] = None,
        processed_hard_root_dir: Optional[Path] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
        processed_transform: Optional[transforms.Compose] = None,
        flip_p: float = 0.0,
        is_val: bool = False,
        is_predict: bool = False,
    ):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.original_root_dir = Path(original_root_dir)
        self.processed_root_dir = Path(processed_root_dir) if processed_root_dir else None
        self.processed_hard_root_dir = Path(processed_hard_root_dir) if processed_hard_root_dir else None
        self.class_to_idx = class_to_idx

        if transform and isinstance(transform, transforms.Compose):
            self.transform = transforms.Compose(
                [t for t in transform.transforms if not isinstance(t, transforms.Normalize)]
            )
        else:
            self.transform = transform

        if processed_transform and isinstance(processed_transform, transforms.Compose):
            self.processed_transform = transforms.Compose(
                [t for t in processed_transform.transforms if not isinstance(t, transforms.Normalize)]
            )
        else:
            self.processed_transform = processed_transform

        self.flip_p = float(flip_p)
        self.is_val = is_val
        self.is_predict = is_predict
        self._default_to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def _open_rgb(self, p: Path) -> Image.Image:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def _to_tensor(self, img: Image.Image, tfm: Optional[transforms.Compose]) -> torch.Tensor:
        x = (tfm or self._default_to_tensor)(img)
        if isinstance(x, dict) and "pixel_values" in x:
            x = x["pixel_values"]
        if not torch.is_tensor(x):
            raise RuntimeError("Transform must yield a torch.Tensor or dict with 'pixel_values'.")
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_name = str(row["img"])
        class_name = str(row.get("classname", ""))

        op = self.original_root_dir / class_name / img_name
        img0 = self._open_rgb(op)
        x0 = self._to_tensor(img0, self.transform)

        if self.is_predict:
            return {"pixel_values": x0, "img_name": img_name}

        assert self.class_to_idx is not None, "class_to_idx must be provided for training/validation"
        label = int(self.class_to_idx[str(row["classname"])])

        if self.is_val:
            return {"pixel_values": x0, "labels": torch.tensor(label, dtype=torch.long)}

        x1 = None
        if self.processed_root_dir:
            p1 = self.processed_root_dir / class_name / f"{Path(img_name).stem}_processed.png"
            if p1.exists():
                img1 = self._open_rgb(p1)
                x1 = self._to_tensor(img1, self.processed_transform or self.transform)
        if x1 is None:
            x1 = x0.clone()

        x2 = None
        if self.processed_hard_root_dir:
            p2 = self.processed_hard_root_dir / class_name / f"{Path(img_name).stem}_processed.png"
            if p2.exists():
                img2 = self._open_rgb(p2)
                x2 = self._to_tensor(img2, self.processed_transform or self.transform)
        if x2 is None:
            x2 = x0.clone()

        # if random.random() < self.flip_p:
        #     x0 = torch.flip(x0, dims=[2])
        #     x1 = torch.flip(x1, dims=[2])
        #     x2 = torch.flip(x2, dims=[2])
        #     if label in FLIP_REMAP:
        #         label = FLIP_REMAP[label]

        return {
            "pixel_values": x0,
            "pixel_values_proc": x1,
            "pixel_values_proc_hard": x2,
            "labels": torch.tensor(label, dtype=torch.long),
        }
