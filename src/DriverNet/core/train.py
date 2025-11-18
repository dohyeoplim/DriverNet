import lightning as L
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.base import BaseModel
from src.DriverNet.utils.logger import wandb_logger

def _build_callbacks(monitor: str, mode: str):
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        verbose=True,
        filename="best-{epoch:02d}-{val/logloss_teacher:.4f}",
        dirpath="output/checkpoints",
    )
    earlyStopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=4,
        min_delta=1e-4,
        check_finite=True,
        verbose=True,
    )
    return [earlyStopping, checkpoint]

def train() -> str:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    model = BaseModel(**cfg.model)
    callbacks = _build_callbacks(monitor="val/logloss", mode="min")

    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger)
    dm = DriverDataModule(**cfg.data)
    trainer.fit(model, datamodule=dm)

    ckpt_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    return ckpt_cb.best_model_path
