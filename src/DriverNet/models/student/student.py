import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy

from src.DriverNet.models.backbone import MODEL_NAMES, MODEL_OPTIONS
from typing import Literal

class Student(L.LightningModule):
    def __init__(
            self,
            name: str = "resnet50",
            num_classes: int = 10,
            pretrained: bool = True,
            lr: float = 3e-4,
            weight_decay: float = 5e-2,
            label_smoothing: float = 0.05,
            scheduler: Literal["onecycle", "cosine", "none"] = "onecycle",
            max_epochs: int = 30,
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
        self.max_epochs: int = max_epochs

        self.criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.train_acc: MulticlassAccuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc: MulticlassAccuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_acc: MulticlassAccuracy = MulticlassAccuracy(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, metric: MulticlassAccuracy, prefix: str):
        x, y = batch["pixel_values"], batch["labels"]
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=-1)
        metric.update(probs, y)
        self.log(f"{prefix}/loss", loss, on_step=True if prefix == "train" else False, on_epoch=True, prog_bar=(prefix != "test"))
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

    def test_step(self, batch, batch_idx):
        self._step(batch, self.test_acc, "test")

    def on_test_epoch_end(self):
        self.log("test/acc", self.test_acc.compute(), on_epoch=True)
        self.test_acc.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["pixel_values"])
        return F.softmax(logits, dim=-1)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt
