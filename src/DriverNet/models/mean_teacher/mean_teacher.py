import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torchmetrics.classification import MulticlassAccuracy
from typing import Literal, Optional

from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.DriverNet.models.backbone import MODEL_OPTIONS
from src.DriverNet.models.base import BaseModel


class MeanTeacherModel(BaseModel):
    def __init__(
        self,
        name: str,
        num_classes: int,
        pretrained: bool,
        lr: float,
        weight_decay: float,
        scheduler: Literal["onecycle", "none"],
        label_smoothing: float = 0.0,
        temperature: float = 0.8,
        cons_weight: float = 0.2,
        cp_weight: float = 0.05,
        max_logit_norm: Optional[float] = None,
        ema_decay: float = 0.996,
    ):
        super().__init__(
            name=name,
            num_classes=num_classes,
            pretrained=pretrained,
            lr=lr,
            weight_decay=weight_decay,
            scheduler=scheduler,
            label_smoothing=label_smoothing,
            temperature=temperature,
            cons_weight=cons_weight,
            cp_weight=cp_weight,
            max_logit_norm=max_logit_norm,
        )

        self.ema_decay: float = float(ema_decay)

        self.teacher = MODEL_OPTIONS[name](num_classes=num_classes, pretrained=pretrained)
        self._init_teacher()

    def _init_teacher(self) -> None:
        self.teacher.load_state_dict(self.model.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def _update_teacher(self) -> None:
        m = self.ema_decay

        for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

        for t_buf, s_buf in zip(self.teacher.buffers(), self.model.buffers()):
            t_buf.data.copy_(s_buf.data)

    @staticmethod
    def _symmetric_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b, dim=-1), reduction="batchmean")
            + F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a, dim=-1), reduction="batchmean")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.teacher(x)
        return logits

    def _step(self, batch, metric: MulticlassAccuracy, prefix: str):
        x_w = batch["pixel_values"]
        x_s = batch.get("pixel_values_proc_hard", x_w)
        y = batch["labels"].long()

        student_logits_w = self._scale(self._maybe_clip_logits(self.model(x_w)))
        student_logits_s = self._scale(self._maybe_clip_logits(self.model(x_s)))

        self.teacher.eval()
        with torch.no_grad():
            teacher_logits_w = self._scale(self._maybe_clip_logits(self.teacher(x_w)))

        student_probs_w = F.softmax(student_logits_w, dim=-1)
        teacher_probs_w = F.softmax(teacher_logits_w, dim=-1)

        ce = self.ce(student_logits_w, y)

        consistency = self._symmetric_kl(student_logits_s, teacher_logits_w)

        confidence_penalty = self._penalty(student_probs_w)

        loss = ce + self.cons_weight * consistency + self.cp_weight * confidence_penalty

        metric.update(teacher_probs_w, y)

        sync_dist = (prefix != "train")
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=sync_dist,
        )

        if prefix == "val":
            val_nll = F.nll_loss(teacher_probs_w.clamp_min(1e-8).log(), y)
            self.log("val/logloss", val_nll, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self._update_teacher()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]

        self.teacher.eval()
        with torch.no_grad():
            logits = self.teacher(x)
            logits = self._maybe_clip_logits(logits)
            probs = F.softmax(logits, dim=1)

        return {
            "preds": probs,
            "img_name": batch["img_name"],
        }
