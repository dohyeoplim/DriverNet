import lightning as L
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.base import BaseModel
from src.DriverNet.utils.logger import wandb_logger

def _build_callbacks(monitor: str, mode: str, fold_idx: int | None = None):
    filename = "best-{epoch:02d}-{val_logloss:.4f}"
    if fold_idx is not None:
        filename = f"fold{fold_idx}-" + filename

    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        verbose=True,
        filename=filename,
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

    num_folds = cfg.data.get("num_folds")

    if num_folds is not None and num_folds > 1:
        best_model_path = ""
        for fold in range(num_folds):
            print(f"Training fold {fold + 1}/{num_folds}")
            model = BaseModel(**cfg.model)
            callbacks = _build_callbacks(monitor="val/logloss", mode="min", fold_idx=fold)

            trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger)

            data_cfg = cfg.data.copy()
            data_cfg["fold_index"] = fold
            dm = DriverDataModule(**data_cfg)

            trainer.fit(model, datamodule=dm)

            ckpt_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
            best_model_path = ckpt_cb.best_model_path
        return best_model_path

    model = BaseModel(**cfg.model)
    callbacks = _build_callbacks(monitor="val/logloss", mode="min")

    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger)
    dm = DriverDataModule(**cfg.data)
    trainer.fit(model, datamodule=dm)

    ckpt_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    return ckpt_cb.best_model_path
