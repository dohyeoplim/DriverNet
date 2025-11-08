import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy

from src.DriverNet.models.backbone import MODEL_NAMES, MODEL_OPTIONS
from typing import Literal

class Student(L.LightningModule):
    def __init__(
            self,
            name: str,
            num_classes: int,
            pretrained: bool,
            lr: float,
            weight_decay: float,
            label_smoothing: float,
            scheduler: Literal["onecycle", "cosine", "none"],
        ):
        super().__init__()
        self.save_hyperparameters()

        if name not in MODEL_NAMES:
            raise ValueError(f"Invalid model: {name}. Available: {MODEL_NAMES}.")
        self.model = MODEL_OPTIONS[name](num_classes=num_classes, pretrained=pretrained)

        self.num_classes: int = num_classes
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.label_smoothing: float = label_smoothing
        self.scheduler_name: Literal["onecycle", "cosine", "none"] = scheduler

        self.criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.train_acc: MulticlassAccuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc: MulticlassAccuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_acc: MulticlassAccuracy = MulticlassAccuracy(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, metric: MulticlassAccuracy, prefix: str):
        x0 = batch["pixel_values"]
        y = batch["labels"]

        logit0 = self(x0)
        ce0 = self.criterion(logit0, y)

        x1 = batch.get("pixel_values_proc")
        if x1 is not None:
            logit1 = self(x1)
            ce1 = self.criterion(logit1, y)

            with torch.no_grad():
                p0 = logit0.softmax(-1)
                p1 = logit1.softmax(-1)

            cons = 0.5 * (F.kl_div(logit0.log_softmax(-1), p1, reduction="batchmean") + F.kl_div(logit1.log_softmax(-1), p0, reduction="batchmean"))
            loss = ce0 + ce1 + 0.5 * cons
            probs_for_metric = 0.5 * (p0 + p1)
        else:
            loss = ce0
            probs_for_metric = logit0.softmax(-1)

        metric.update(probs_for_metric, y)
        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_acc, "train")

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), on_epoch=True, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.val_acc, "val")

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    # def test_step(self, batch, batch_idx):
    #     self._step(batch, self.test_acc, "test")

    # def on_test_epoch_end(self):
    #     self.log("test/acc", self.test_acc.compute(), on_epoch=True)
    #     self.test_acc.reset()

    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        y_hat = self(x)
        probs = F.softmax(y_hat, dim=1)
        return {"preds": probs, "img_name": batch["img_name"]}

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt
