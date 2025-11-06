import lightning as L
import torch
import torch.nn as nn
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.teacher import Teacher
from typing import List, cast
from pathlib import Path

def main():
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    test_data_dir = "./input/imgs/test"

    dm = DriverDataModule(**cfg.data, test_dir=test_data_dir)
    model = Teacher(**cfg.model.teacher)
    trainer = L.Trainer(**cfg.trainer)

    dm.setup(stage="test")

    raw_predictions = trainer.predict(model, dataloaders=dm.test_dataloader())
    assert raw_predictions is not None

    predictions_list = cast(List[torch.Tensor], raw_predictions)

    all_preds_tensor = torch.cat(predictions_list, dim=0)
    all_preds_np = all_preds_tensor.cpu().numpy()

    img_names = []
    for batch in dm.test_dataloader():
        img_names.extend(batch["img_name"])

    num_classes_layer = model.model.heads.head
    assert isinstance(num_classes_layer, nn.Linear)
    num_classes = num_classes_layer.out_features
    submission_df = pd.DataFrame(all_preds_np, columns=[f'c{i}' for i in range(num_classes)])
    submission_df.insert(0, 'img', img_names)

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    submission_path = output_dir / "submission_data.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission created at {submission_path}")

if __name__ == "__main__":
    main()
