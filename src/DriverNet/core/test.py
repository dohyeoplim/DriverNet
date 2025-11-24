import lightning as L
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.base import BaseModel
from src.DriverNet.utils.submission import create_submission

def test(checkpoint_path: str | None = None, submission_path: str = "./output/submission.csv") -> None:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    if checkpoint_path:
        model = BaseModel.load_from_checkpoint(checkpoint_path)
    else:
        model = BaseModel(**cfg.model)

    dm = DriverDataModule(**cfg.data_test)

    dm.setup(stage="predict")

    trainer = L.Trainer(**cfg.trainer)

    outputs = trainer.predict(model, datamodule=dm)

    create_submission(outputs, submission_path) # type: ignore
