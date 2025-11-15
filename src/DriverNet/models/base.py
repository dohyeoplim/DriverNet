import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torchmetrics.classification import MulticlassAccuracy
from typing import Literal, Optional
import math

from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.DriverNet.models.backbone import MODEL_NAMES, MODEL_OPTIONS
from src.DriverNet.utils.ema import EMA
from src.DriverNet.utils.losses import (
    kl_div_with_temperature,
    get_hard_example_weights,
    entropy_gated_kd,
)

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
        cons_weight: float = 0.2,
        ce_x1_weight: float = 0.5,
        max_logit_norm: Optional[float] = None,
        ema_decay: float = 0.996,
        ema_warmup_steps: int = 100,
        teacher_entropy_weight: float = 1.0,
        student_error_weight: float = 1.0,
        kd_temperature: float = 2.0,
        kd_x1_weight: float = 1.0,
        kd_x2_weight: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.name = name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.label_smoothing = label_smoothing
        self.cons_weight = cons_weight
        self.ce_x1_weight = ce_x1_weight
        self.max_logit_norm = max_logit_norm
        self.ema_decay = ema_decay
        self.ema_warmup_steps = ema_warmup_steps
        self.teacher_entropy_weight = teacher_entropy_weight
        self.student_error_weight = student_error_weight
        self.kd_temperature = kd_temperature
        self.kd_x1_weight = kd_x1_weight
        self.kd_x2_weight = kd_x2_weight

        if self.name not in MODEL_NAMES:
            raise ValueError(f"Invalid model: {self.name}. Available: {MODEL_NAMES}.")
        self.model = MODEL_OPTIONS[self.name](num_classes=self.num_classes, pretrained=self.pretrained)

        self.teacher_ema = EMA(self.model, decay=self.ema_decay, warmup_steps=self.ema_warmup_steps)
        self.teacher = self.teacher_ema.teacher

        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc_student = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc_teacher = MulticlassAccuracy(num_classes=self.num_classes)

        self._max_entropy = math.log(self.num_classes)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.teacher_ema.update()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher(x)

    def _maybe_clip_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.max_logit_norm is not None:
            with torch.no_grad():
                n = logits.norm(dim=1, keepdim=True).clamp_min(1e-6)
                s = (self.max_logit_norm / n).clamp_max(1.0)
            logits = logits * s
        return logits

    def _step(self, batch, prefix: str):
        x0 = batch["pixel_values"]
        x1 = batch.get("pixel_values_proc", x0)
        x2 = batch.get("pixel_values_proc_hard", x0)
        y = batch["labels"].long()

        def fwd(model, x):
            return self._maybe_clip_logits(model(x))

        student_logits_x0 = fwd(self.model, x0)
        student_logits_x1 = fwd(self.model, x1)
        student_logits_x2 = fwd(self.model, x2)

        with torch.no_grad():
            teacher_logits_x0 = fwd(self.teacher, x0)

        sample_weights = get_hard_example_weights(
            student_logits_x0,
            teacher_logits_x0,
            y,
            self.num_classes,
            self.teacher_entropy_weight,
            self.student_error_weight,
            self.label_smoothing,
        )

        ce_x0_per_sample = F.cross_entropy(student_logits_x0, y, label_smoothing=self.label_smoothing, reduction='none')
        ce_x0 = (ce_x0_per_sample * sample_weights).mean()
        ce_x1 = F.cross_entropy(student_logits_x1, y, label_smoothing=self.label_smoothing)
        ce_loss = ce_x0 + self.ce_x1_weight * ce_x1

        kd_x0 = kl_div_with_temperature(student_logits_x0, teacher_logits_x0, self.kd_temperature)
        kd_x1 = kl_div_with_temperature(student_logits_x1, teacher_logits_x0, self.kd_temperature)

        with torch.no_grad():
            teacher_probs_x0 = F.softmax(teacher_logits_x0, dim=-1)
            teacher_entropy = -torch.sum(teacher_probs_x0 * torch.log(teacher_probs_x0.clamp_min(1e-8)), dim=1)
            normalized_entropy = teacher_entropy / self._max_entropy

        kd_x2 = entropy_gated_kd(student_logits_x2, teacher_logits_x0, self.kd_temperature, normalized_entropy)

        consistency_loss = (
            kd_x0 + self.kd_x1_weight * kd_x1 + self.kd_x2_weight * kd_x2
        ) / (1 + self.kd_x1_weight + self.kd_x2_weight)

        loss = ce_loss + self.cons_weight * consistency_loss

        sync_dist = prefix != "train"
        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True, sync_dist=sync_dist)
        self.log(f"{prefix}/ce_loss", ce_loss, on_step=False, on_epoch=True, sync_dist=sync_dist)
        self.log(f"{prefix}/consistency_loss", consistency_loss, on_step=False, on_epoch=True, sync_dist=sync_dist)

        if prefix == "train":
            self.train_acc.update(F.softmax(student_logits_x0, dim=-1), y)
        elif prefix == "val":
            student_probs = F.softmax(student_logits_x0, dim=-1)
            teacher_probs = F.softmax(teacher_logits_x0, dim=-1)
            self.val_acc_student.update(student_probs, y)
            self.val_acc_teacher.update(teacher_probs, y)

            val_nll = F.nll_loss((teacher_probs.clamp_min(1e-8)).log(), y)
            self.log("val/logloss", val_nll, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc_student", self.val_acc_student.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc_teacher", self.val_acc_teacher.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_acc_student.reset()
        self.val_acc_teacher.reset()

    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        logits = self.teacher_forward(x)
        logits = self._maybe_clip_logits(logits)
        probs = F.softmax(logits, dim=1)
        return {"preds": probs, "img_name": batch["img_name"]}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler == "none":
            return opt

        if self.scheduler == "onecycle":
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
