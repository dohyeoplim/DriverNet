import lightning as L
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.base import BaseModel
from src.DriverNet.utils.logger import wandb_logger
from src.DriverNet.core.test import test

def _build_callbacks(monitor: str, mode: str, fold_idx: int | None = None) -> list[L.Callback]:
    earlyStopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=4,
        min_delta=1e-4,
        check_finite=True,
        verbose=True,
    )
    return [earlyStopping]

def train() -> list[str]:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    num_folds = cfg.data.get("num_folds")

    if num_folds is not None and num_folds > 1:
        submission_paths = []
        for fold in range(num_folds):
            print(f"Training fold {fold + 1}/{num_folds}")
            model = BaseModel(**cfg.model)
            callbacks = _build_callbacks(monitor="val/logloss", mode="min", fold_idx=fold)

            trainer = L.Trainer(**cfg.trainer, log_every_n_steps=10, callbacks=callbacks, logger=wandb_logger)

            data_cfg = cfg.data.copy()
            data_cfg["fold_index"] = fold
            dm = DriverDataModule(**data_cfg)

            trainer.fit(model, datamodule=dm)

            submission_path = f"./output/submission_fold_{fold}.csv"
            test(model, submission_path=submission_path)
            submission_paths.append(submission_path)

        return submission_paths

    model = BaseModel(**cfg.model)
    callbacks = _build_callbacks(monitor="val/logloss", mode="min")

    trainer = L.Trainer(**cfg.trainer, log_every_n_steps=20, callbacks=callbacks, logger=wandb_logger)
    dm = DriverDataModule(**cfg.data)
    trainer.fit(model, datamodule=dm)

    test(model)
    return ["./output/submission.csv"]
