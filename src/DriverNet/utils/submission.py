import pandas as pd
import torch
from src.DriverNet.data.DataModule import DriverDataModule
from typing import Any

def create_submission(outputs, path="submission.csv"):
    preds = torch.cat([o["preds"] for o in outputs], dim=0).cpu().numpy()
    names = sum([o["img_name"] for o in outputs], [])
    assert len(names) == preds.shape[0], "Mismatch between names and predictions."

    df = pd.DataFrame(preds, columns=[f"c{i}" for i in range(preds.shape[1])])
    df.insert(0, "img", names)
    df.to_csv(path, index=False)
    print(f"✅ Saved submission: {path} ({len(df)} rows)")
