import lightning as L
from omegaconf import OmegaConf, DictConfig
from src.DriverNet.data.DataModule import DriverDataModule
from src.DriverNet.models.base import BaseModel
from src.DriverNet.utils.submission import create_submission

def test(model: BaseModel | None = None, checkpoint_path: str | None = None, submission_path: str = "./output/submission.csv") -> None:
    cfg = OmegaConf.load("configs/config.yaml")
    assert isinstance(cfg, DictConfig)

    if checkpoint_path is not None:
        model = BaseModel.load_from_checkpoint(checkpoint_path=checkpoint_path, **cfg.model)
    elif model is None:
        raise ValueError("Either 'model' or 'checkpoint_path' must be provided.")

    model.eval()
    model.freeze()

    dm = DriverDataModule(**cfg.data_test)

    dm.setup(stage="predict")

    trainer = L.Trainer(
        **cfg.trainer,
        logger=False,
    )

    trainer.predict(model, datamodule=dm, return_predictions=False)

    if trainer.global_rank == 0:
        create_submission(submission_path)
