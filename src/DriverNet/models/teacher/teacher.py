import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler
from torchmetrics.classification import MulticlassAccuracy
from typing import Literal, Optional

from src.DriverNet.models.backbone import MODEL_NAMES, MODEL_OPTIONS


##### TODO: 코드 리팩토링 #####
class Teacher(L.LightningModule):
    def __init__(
        self,
        name: str,
        num_classes: int,
        pretrained: bool,
        lr: float,
        weight_decay: float,
        scheduler: Literal["onecycle", "none"],
        cp_weight: float = 0.05,
        max_logit_norm: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if name not in MODEL_NAMES:
            raise ValueError(f"Invalid model: {name}. Available: {MODEL_NAMES}.")
        self.model = MODEL_OPTIONS[name](num_classes=num_classes, pretrained=pretrained)

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name: Literal["onecycle", "none"] = scheduler

        self.cp_weight: float = float(cp_weight)
        self.max_logit_norm: Optional[float] = max_logit_norm

        self.ce = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _maybe_clip_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.max_logit_norm is not None:
            with torch.no_grad():
                n = logits.norm(dim=1, keepdim=True).clamp_min(1e-6)
                s = (self.max_logit_norm / n).clamp_max(1.0)
            logits = logits * s
        return logits

    @staticmethod
    def _penalty(probs: torch.Tensor) -> torch.Tensor:
        return (probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def _step(self, batch, metric: MulticlassAccuracy, prefix: str):
        x = batch["pixel_values"]
        y = batch["labels"].long()

        logits = self._maybe_clip_logits(self(x))
        ce = self.ce(logits, y)
        probs = F.softmax(logits, dim=-1)
        cp = self._penalty(probs)

        loss = ce + self.cp_weight * cp

        metric.update(probs, y)
        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True, sync_dist=(prefix != "train"))

        if prefix == "val":
            self.log("val/logloss", ce, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_acc, "train")

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.val_acc, "val")

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        y = batch["labels"].long()
        logits = self._maybe_clip_logits(self(x))
        ce = self.ce(logits, y)
        probs = F.softmax(logits, dim=-1)
        self.test_acc.update(probs, y)
        self.log("test/logloss", ce, on_epoch=True, prog_bar=True, sync_dist=True)
        return ce

    def on_test_epoch_end(self):
        self.log("test/acc", self.test_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_acc.reset()

    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        img_names = batch["img_name"]
        logits = self._maybe_clip_logits(self(x))
        probs = F.softmax(logits, dim=1)
        return {"preds": probs, "img_name": img_names}

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_name == "none":
            return opt

        if self.scheduler_name == "onecycle":
            if self.trainer is None:
                raise RuntimeError("Trainer not attached; OneCycle needs total steps.")

            total_steps = int(self.trainer.estimated_stepping_batches)
            sched: LRScheduler = OneCycleLR(
                opt,
                max_lr=self.lr,
                total_steps=max(1, total_steps),
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1e4,
                anneal_strategy="cos",
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        return opt