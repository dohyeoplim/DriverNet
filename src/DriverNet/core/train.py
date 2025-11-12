import lightning as L
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.teacher import Teacher
from src.DriverNet.models.student import Student
from src.DriverNet.utils.logger import wandb_logger


def _build_callbacks(monitor: str, mode: str, prefix: str):
    metric_token = monitor.replace("/", "_")
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        verbose=True,
        filename=f"best-{prefix}" + "-{epoch:02d}-{" + f"{metric_token}:.4f" + "}",
        dirpath="output/checkpoints",
    )
    earlyStopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=3,
        min_delta=1e-4,
        check_finite=True,
        verbose=True,
    )
    return [earlyStopping, checkpoint]

def train(what: str) -> str:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    if what == "teacher":
        model = Teacher(**cfg.model.teacher)
        callbacks = _build_callbacks(monitor="val/acc", mode="max", prefix="teacher")
    elif what == "student":
        model = Student(**cfg.model.student)
        callbacks = _build_callbacks(monitor="val/logloss", mode="min", prefix="student")
    else:
        raise ValueError(f"Unknown model type: {what}")

    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger)
    dm = DriverDataModule(**cfg.data)
    trainer.fit(model, datamodule=dm)

    ckpt_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    return ckpt_cb.best_model_path
