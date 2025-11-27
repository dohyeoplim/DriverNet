import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Sequence

def create_submission(path: str = "submission.csv") -> None:
    out_dir = Path("output/predictions")
    shard_paths = sorted(out_dir.glob("preds_rank*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No prediction shards found in {out_dir}")

    all_preds = []
    all_names: list[str] = []

    for shard_path in shard_paths:
        data = torch.load(shard_path, map_location="cpu")
        preds_t: torch.Tensor = data["preds"]
        names: Sequence[str] = data["img_name"]

        all_preds.append(preds_t)
        all_names.extend(list(names))

    preds_t = torch.cat(all_preds, dim=0).detach().cpu()
    preds: np.ndarray = preds_t.numpy()
    preds = np.clip(preds, 1e-15, 1 - 1e-15)

    if preds.shape[0] != len(all_names):
        raise ValueError(f"Mismatch: names={len(all_names)} vs preds={preds.shape[0]}")

    columns = pd.Index([f"c{i}" for i in range(preds.shape[1])], dtype="object")
    df = pd.DataFrame(data=preds, columns=columns, copy=False)
    df.insert(0, "img", pd.Series(all_names, dtype="string"))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved submission: {path} ({len(df)} rows)")
