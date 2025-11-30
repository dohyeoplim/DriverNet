import numpy as np
import pandas as pd
import torch
from src.DriverNet.data.DataModule import DriverDataModule
from typing import Any, Sequence, TypedDict

class StepOut(TypedDict):
    preds: torch.Tensor
    img_name: Sequence[str]

def create_submission(outputs: Sequence[StepOut], path="submission.csv"):
    preds_t = torch.cat([o["preds"] for o in outputs], dim=0).detach().cpu()
    preds: np.ndarray = preds_t.numpy()
    preds = np.clip(preds, 1e-15, 1 - 1e-15)

    names: list[str] = [n for o in outputs for n in o["img_name"]]

    if preds.shape[0] != len(names):
        raise ValueError(f"Mismatch: names={len(names)} vs preds={preds.shape[0]}")

    columns = pd.Index([f"c{i}" for i in range(preds.shape[1])], dtype="object")
    df = pd.DataFrame(data=preds, columns=columns, copy=False)

    df.insert(0, "img", pd.Series(names, dtype="string"))

    df.to_csv(path, index=False)
    print(f"✅ Saved submission: {path} ({len(df)} rows)")
