import lightning as L
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.teacher import Teacher
from finetuning_scheduler import FinetuningScheduler

def train(what: str) -> None:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    if what == "teacher":
        model = Teacher(**cfg.model.teacher)
        trainer = L.Trainer(**cfg.trainer, callbacks=[FinetuningScheduler()])
        trainer.ckpt_path = "output/teacher/checkpoints/best.ckpt"
        dm = DriverDataModule(**cfg.data)
        trainer.fit(model, datamodule=dm)

    elif what == "student":
        pass
    else:
        raise ValueError(f"Unknown model type: {what}")
