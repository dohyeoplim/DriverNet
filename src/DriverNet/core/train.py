import lightning as L
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.teacher import Teacher
from src.DriverNet.models.student import Student
from finetuning_scheduler import FinetuningScheduler
from finetuning_scheduler.fts_supporters import FTSEarlyStopping, FTSCheckpoint
import torch

def train(what: str) -> None:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    if what == "teacher":
        model = Teacher(**cfg.model.teacher)
        callbacks = [
            FinetuningScheduler(),
            FTSEarlyStopping(monitor="val_acc", mode="max"),
            FTSCheckpoint(monitor="val_acc", mode="max", save_top_k=1, verbose=True, filename='best-{epoch:02d}-{val_acc:.4f}')
        ]
        trainer = L.Trainer(**cfg.trainer, callbacks=callbacks)
        dm = DriverDataModule(**cfg.data)
        trainer.fit(model, datamodule=dm)

    elif what == "student":
        model = Student(**cfg.model.student)
        trainer = L.Trainer(**cfg.trainer)
        dm = DriverDataModule(**cfg.data)
        trainer.fit(model, datamodule=dm)

    else:
        raise ValueError(f"Unknown model type: {what}")
