import random
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import torch
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
        original_root_dir: str | Path,
        processed_root_dir: Optional[str | Path] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
        flip_p: float = 0.5,
        is_predict: bool = False,
    ):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.original_root_dir = Path(original_root_dir)
        if processed_root_dir is not None:
            self.processed_root_dir = Path(processed_root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.flip_p = float(flip_p)
        self.is_predict = is_predict

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = row["img"]
        class_name = row["classname"] if "classname" in row else ""
        original_img_path = self.original_root_dir / class_name / img_name
        processed_image_path = self.processed_root_dir / class_name / img_name.replace(".jpg", "_processed.png")

        original_image = Image.open(original_img_path).convert("RGB")
        processed_image = Image.open(processed_image_path).convert("RGB")

        if self.transform is not None:
            original_image = self.transform(original_image)

        assert isinstance(original_image, torch.Tensor)
        assert isinstance(processed_image, torch.Tensor)

        if self.is_predict:
            return {"pixel_values": original_image, "img_name": img_name}

        assert self.class_to_idx is not None
        label = self.class_to_idx[row["classname"]]

        # if random.random() < self.flip_p:
        #     image = torch.flip(original_image, dims=[2])
        #     if label in FLIP_REMAP:
        #         label = FLIP_REMAP[label]

        return {
            "pixel_values": original_image,
            "pixel_values_proc": processed_image,
            "labels": torch.tensor(label, dtype=torch.long),
        }
