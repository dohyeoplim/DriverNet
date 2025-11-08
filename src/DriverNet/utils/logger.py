from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="DriverNet", entity="dohyeoplim-edu", log_model="all")
