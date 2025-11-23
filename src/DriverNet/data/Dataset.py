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
        depth_root_dir: Optional[Path] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        flip_p: float = 0.0,
        is_predict: bool = False,
    ):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.original_root_dir = Path(original_root_dir)
        self.depth_root_dir = Path(depth_root_dir) if depth_root_dir else None
        self.class_to_idx = class_to_idx
        self.flip_p = float(flip_p)
        self.is_predict = is_predict
        self._default_to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def _open_rgb(self, p: Path) -> Image.Image:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def _open_grayscale(self, p: Path) -> Image.Image:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Depth image not found at {p}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(img)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        return self._default_to_tensor(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_name = str(row["img"])
        if not self.is_predict:
            class_name = str(row.get("classname", ""))

        op = self.original_root_dir / class_name / img_name
        img0 = self._open_rgb(op)
        x0 = self._to_tensor(img0)

        assert self.class_to_idx is not None, "class_to_idx must be provided for training/validation"
        label = int(self.class_to_idx[str(row["classname"])])

        if self.depth_root_dir:
            p1 = self.depth_root_dir / class_name / f"{Path(img_name).stem}_depth.png"
            if p1.exists():
                img1 = self._open_grayscale(p1)
                xd = self._to_tensor(img1)
        if xd is None:
            raise FileNotFoundError(f"Depth image not found for {img_name} at {p1}")

        # if random.random() < self.flip_p:
        #     x0 = torch.flip(x0, dims=[2])
        #     x1 = torch.flip(x1, dims=[2])
        #     x2 = torch.flip(x2, dims=[2])
        #     if label in FLIP_REMAP:
        #         label = FLIP_REMAP[label]

        if self.is_predict:
            return {
                "pixel_values": x0,
                "depth": xd,
                "img_name": row["img"],
            }

        return {
            "pixel_values": x0,
            "depth": xd,
            "labels": torch.tensor(label, dtype=torch.long),
        }
