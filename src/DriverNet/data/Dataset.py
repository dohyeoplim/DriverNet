import random
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FLIP_REMAP = {
    1: 3,  # c1(texting-R) <-> c3(texting-L)
    3: 1,
    2: 4,  # c2(phone-R) <-> c4(phone-L)
    4: 2,
}

class DriverDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        original_root_dir: Path,
        processed_root_dir: Optional[Path] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
        processed_transform: Optional[transforms.Compose] = None,
        flip_p: float = 0.0,
        is_predict: bool = False,
    ):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.original_root_dir = original_root_dir
        self.processed_root_dir = processed_root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.processed_transform = processed_transform
        self.flip_p = float(flip_p)
        self.is_predict = is_predict

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_name = row["img"]
        class_name = row.get("classname", "")

        original_img_path = self.original_root_dir / class_name / img_name
        original_image = Image.open(original_img_path).convert("RGB")

        if self.transform:
            original_image = self.transform(original_image)

        if self.is_predict:
            return {"pixel_values": original_image, "img_name": img_name}

        assert self.class_to_idx is not None, "class_to_idx must be provided for training/validation"
        label = self.class_to_idx[row["classname"]]

        processed_image = None
        if self.processed_root_dir:
            processed_image_path = self.processed_root_dir / class_name / f"{Path(img_name).stem}_processed.png"
            if processed_image_path.exists():
                processed_image = Image.open(processed_image_path).convert("RGB")
                if self.processed_transform:
                    processed_image = self.processed_transform(processed_image)

        has_proc = processed_image is not None

        if random.random() < self.flip_p:
            assert isinstance(original_image, torch.Tensor)
            original_image = torch.flip(original_image, dims=[2])
            if has_proc:
                assert isinstance(processed_image, torch.Tensor)
                processed_image = torch.flip(processed_image, dims=[2])

            if label in FLIP_REMAP:
                label = FLIP_REMAP[label]

        if not has_proc:
            assert isinstance(original_image, torch.Tensor)
            processed_image = original_image.clone()

        return {
            "pixel_values": original_image,
            "pixel_values_proc": processed_image,
            "has_proc": torch.tensor(has_proc),
            "labels": torch.tensor(label, dtype=torch.long),
        }
