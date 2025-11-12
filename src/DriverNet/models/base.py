import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torchmetrics.classification import MulticlassAccuracy
from typing import Literal, Optional

from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.DriverNet.models.backbone import MODEL_NAMES, MODEL_OPTIONS

class BaseModel(L.LightningModule):
    def __init__(
        self,
        name: str,
        num_classes: int,
        pretrained: bool,
        lr: float,
        weight_decay: float,
        scheduler: Literal["onecycle", "none"],
        label_smoothing: float = 0,
        temperature: float = 0.8,
        cons_weight: float = 0.2,
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

        self.label_smoothing: float = float(label_smoothing)
        self.temperature: float = float(temperature)
        self.cons_weight: float = float(cons_weight)
        self.cp_weight: float = float(cp_weight)
        self.max_logit_norm: Optional[float] = max_logit_norm

        self.ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

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

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        return logits if T == 1.0 else logits / T

    @staticmethod
    def _penalty(probs: torch.Tensor) -> torch.Tensor:
        return (probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def _step(self, batch, metric: MulticlassAccuracy, prefix: str):
        x0 = batch["pixel_values"]
        x1 = batch.get("pixel_values_proc", x0)
        x2 = batch.get("pixel_values_proc_hard", x0)
        y = batch["labels"].long()

        def fwd(x):
            return self._scale(self._maybe_clip_logits(self(x)))

        logits = [fwd(x0), fwd(x1), fwd(x2)]
        probs  = [F.softmax(l, dim=-1) for l in logits]

        ce_terms = [self.ce(l, y) for l in logits]
        ce = torch.stack(ce_terms).mean()

        def symmetric_kl(a, b):
            return 0.5 * (F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b, dim=-1), reduction="batchmean") + F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a, dim=-1), reduction="batchmean"))

        consistency = (symmetric_kl(logits[0], logits[1]) + symmetric_kl(logits[0], logits[2]) + symmetric_kl(logits[1], logits[2])) / 3.0

        probs_avg = torch.stack(probs, dim=0).mean(dim=0)
        confidence_penalty = self._penalty(probs_avg)

        loss = ce + self.cons_weight * consistency + self.cp_weight * confidence_penalty

        metric.update(probs_avg, y)
        sync_dist = (prefix != "train")
        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True, sync_dist=sync_dist)

        if prefix == "val":
            # base_logits = torch.stack(logits, dim=0).mean(dim=0)
            # self.log("val/logloss", self.ce(base_logits, y), on_epoch=True, prog_bar=True, sync_dist=True)
            p = (F.softmax(logits[0], dim=-1) + F.softmax(logits[1], dim=-1) + F.softmax(logits[2], dim=-1)) / 3
            val_nll = F.nll_loss((p.clamp_min(1e-8)).log(), y)
            self.log("val/logloss", val_nll, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_acc, "train")

        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]

        if hasattr(opt, "param_groups") and len(opt.param_groups) > 0:
            lr = opt.param_groups[0].get("lr", None)
            if lr is not None:
                self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)

        return loss

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.val_acc, "val")

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_acc.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
                pct_start=0.25,
                div_factor=25.0,
                final_div_factor=1e3,
                anneal_strategy="cos",
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        return opt
