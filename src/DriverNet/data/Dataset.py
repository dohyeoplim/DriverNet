import random
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

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
        else:
            self.processed_root_dir = None
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.flip_p = float(flip_p)
        self.is_predict = is_predict

    ########## TODO: 임시 ##########
    @staticmethod
    def _to_tensor_norm(img) -> torch.Tensor:
        tf = transforms.Compose([
            transforms.Resize(int(224 * 1.14), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return tf(img)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = row["img"]
        class_name = row["classname"] if "classname" in row else ""
        original_img_path = self.original_root_dir / class_name / img_name
        original_image = Image.open(original_img_path).convert("RGB")

        if self.processed_root_dir is not None:
            processed_image_path = self.processed_root_dir / class_name / f"{Path(img_name).stem}_processed.png"
            if processed_image_path.exists():
                processed_image = Image.open(processed_image_path).convert("RGB")
            else:
                processed_image = None
        else:
            processed_image = None

        if self.transform is not None:
            original_image = self.transform(original_image)

        if processed_image is not None:
            processed_image = self._to_tensor_norm(processed_image)

        if self.is_predict:
            return {"pixel_values": original_image, "img_name": img_name} # type: ignore

        assert self.class_to_idx is not None
        label = self.class_to_idx[row["classname"]]

        # if random.random() < self.flip_p:
        #     image = torch.flip(original_image, dims=[2])
        #     if label in FLIP_REMAP:
        #         label = FLIP_REMAP[label]

        has_proc = processed_image is not None
        if not has_proc:
            processed_image = original_image.clone() # type: ignore

        return {
            "pixel_values": original_image, # type: ignore
            "pixel_values_proc": processed_image,
            "has_proc": torch.tensor(has_proc),
            "labels": torch.tensor(label, dtype=torch.long),
        }
