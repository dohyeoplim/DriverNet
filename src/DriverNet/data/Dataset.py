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
        root_dir: str | Path,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        flip_p: float = 0.5,
    ):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.flip_p = float(flip_p)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["classname"] / row["img"]

        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["classname"]]

        if self.transform is not None:
            image = self.transform(image)

        assert isinstance(image, torch.Tensor)

        if random.random() < self.flip_p:
            image = torch.flip(image, dims=[2])
            if label in FLIP_REMAP:
                label = FLIP_REMAP[label]

        return {
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long),
        }
