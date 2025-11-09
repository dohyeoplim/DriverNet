import lightning as L
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.teacher import Teacher
from src.DriverNet.models.student import Student
from src.DriverNet.utils.submission import create_submission

def test(what: str, checkpoint_path: str | None = None) -> None:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    if what == "teacher":
        if checkpoint_path:
            model = Teacher.load_from_checkpoint(checkpoint_path)
        else:
            model = Teacher(**cfg.model.teacher)
    elif what == "student":
        if checkpoint_path:
            model = Student.load_from_checkpoint(checkpoint_path)
        else:
            model = Student(**cfg.model.student)
    else:
        raise ValueError(f"Unknown model type: {what!r}")

    dm = DriverDataModule(**cfg.data_test)

    dm.setup(stage="predict")

    trainer = L.Trainer(**cfg.trainer)

    outputs = trainer.predict(model, datamodule=dm)

    create_submission(outputs, "./output/submission.csv")
