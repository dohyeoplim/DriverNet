import lightning as L
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.teacher import Teacher
from src.DriverNet.models.student import Student
from src.DriverNet.utils.logger import wandb_logger
from lightning.pytorch.callbacks import ModelCheckpoint
from finetuning_scheduler import FinetuningScheduler
from finetuning_scheduler.fts_supporters import FTSEarlyStopping, FTSCheckpoint

def train(what: str) -> str:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    if what == "teacher":
        model = Teacher(**cfg.model.teacher)
        checkpoint_callback = FTSCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            verbose=True,
            filename='best-teacher-{epoch:02d}-{val_acc:.4f}',
            dirpath="output/checkpoints"
        )
        callbacks = [
            FinetuningScheduler(),
            FTSEarlyStopping(monitor="val_acc", mode="max"),
            checkpoint_callback
        ]
        trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger)
        dm = DriverDataModule(**cfg.data)
        trainer.fit(model, datamodule=dm)
        return checkpoint_callback.best_model_path

    elif what == "student":
        model = Student(**cfg.model.student)
        checkpoint_callback = ModelCheckpoint(
            monitor="val/logloss",
            mode="min",
            save_top_k=1,
            verbose=True,
            filename='best-student-{epoch:02d}-{val/logloss:.4f}',
            dirpath="output/checkpoints"
        )
        trainer = L.Trainer(**cfg.trainer, logger=wandb_logger, callbacks=[checkpoint_callback])
        dm = DriverDataModule(**cfg.data)
        trainer.fit(model, datamodule=dm)
        return checkpoint_callback.best_model_path

    else:
        raise ValueError(f"Unknown model type: {what}")
