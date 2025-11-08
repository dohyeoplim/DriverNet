import lightning as L
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.teacher import Teacher
from src.DriverNet.models.student import Student

def test(what: str) -> None:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    test_data_dir = "./input/imgs/test"

    dm = DriverDataModule(**cfg.data, test_dir=test_data_dir)
    model = Teacher(**cfg.model.teacher)
    trainer = L.Trainer(**cfg.trainer)

    dm.setup(stage="test")

    if what == "teacher":
        model = Teacher(**cfg.model.teacher)
        trainer = L.Trainer(**cfg.trainer)
        dm = DriverDataModule(**cfg.data)
        trainer.test(model, datamodule=dm)

    elif what == "student":
        model = Student(**cfg.model.student)
        trainer = L.Trainer(**cfg.trainer)
        dm = DriverDataModule(**cfg.data)
        trainer.test(model, datamodule=dm)

    else:
        raise ValueError(f"Unknown model type: {what}")
