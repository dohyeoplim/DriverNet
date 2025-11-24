import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from typing import Literal, Optional
import math

from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.DriverNet.data.Transforms import Augmentations
from src.DriverNet.models.backbone import MODEL_NAMES, MODEL_OPTIONS
from src.DriverNet.utils.ema import EMA
from src.DriverNet.utils.visualization import save_confusion_matrix

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
        consistency_loss_type: Literal["kd", "mse"] = "kd",
        consistency_rampup_steps: float = 0.2,
        final_consistency_weight: float = 1.0,
        max_logit_norm: Optional[float] = None,
        ema_decay: float = 0.999,
        ema_warmup_steps: int = 100,
        augment_on_gpu: bool = True,
        image_size: int = 224,
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
        self.consistency_loss_type = consistency_loss_type
        self.consistency_rampup_steps = consistency_rampup_steps
        self.final_consistency_weight = final_consistency_weight
        self.max_logit_norm = max_logit_norm
        self.ema_decay = ema_decay
        self.ema_warmup_steps = ema_warmup_steps
        self.augment_on_gpu = augment_on_gpu
        self.image_size = image_size

        if self.name not in MODEL_NAMES:
            raise ValueError(f"Invalid model: {self.name}. Available: {MODEL_NAMES}.")
        self.model = MODEL_OPTIONS[self.name](num_classes=self.num_classes, pretrained=self.pretrained)
        model_name_lower = self.name.lower()
        self._uses_depth: bool = ("depthg" in model_name_lower) or ("depthgrouped" in type(self.model).__name__.lower())

        self.teacher_ema = EMA(self.model, decay=self.ema_decay, warmup_steps=self.ema_warmup_steps)
        self.teacher = self.teacher_ema.teacher

        self._consistency_weight = 0.0

        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc_student = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc_teacher = MulticlassAccuracy(num_classes=self.num_classes)

        self.val_confmat_student = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.val_confmat_teacher = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self._final_confmat_student: Optional[torch.Tensor] = None
        self._final_confmat_teacher: Optional[torch.Tensor] = None

        self._augmentations = Augmentations(self.image_size)

        self._IMGNET_NORMALIZE = K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._max_entropy = math.log(self.num_classes)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.teacher_ema.update()

        start = self.ema_warmup_steps
        end = start + self.consistency_rampup_steps

        if self.global_step < start:
            self._consistency_weight = 0.0
        elif start <= self.global_step < end:
            ramp_pos = (self.global_step - start) / (end - start)
            self._consistency_weight = self.final_consistency_weight * ramp_pos
        else:
            self._consistency_weight = self.final_consistency_weight

        self.log("consistency_weight", self._consistency_weight, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._uses_depth:
            if depth is None:
                raise ValueError("Depth map is required")
            return self.model(x, depth)
        return self.model(x)

    @torch.no_grad()
    def teacher_forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._uses_depth:
            if depth is None:
                raise ValueError("Depth map is required")
            return self.teacher(x, depth)
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
        depth = batch.get("depth") if self._uses_depth else None
        y = batch["labels"].long()

        if self.training:
            x0 = self._augmentations(x0)
            x0 = self._IMGNET_NORMALIZE(x0)
        else:
            x0 = self._IMGNET_NORMALIZE(x0)

        def fwd(m: nn.Module, x: torch.Tensor, d: Optional[torch.Tensor]) -> torch.Tensor:
            if self._uses_depth:
                logits = m(x, d)
            else:
                logits = m(x)
            return self._maybe_clip_logits(logits)

        student_logits_x0 = fwd(self.model, x0, depth)
        with torch.no_grad():
            teacher_logits_x0 = fwd(self.teacher, x0, depth)

        ce_loss = F.cross_entropy(student_logits_x0, y, label_smoothing=self.label_smoothing)

        if self.consistency_loss_type == "kd":
            def loss_fn(s_logits, t_logits):
                return F.kl_div(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits.detach(), dim=-1), reduction="batchmean")
        elif self.consistency_loss_type == "mse":
            def loss_fn(s_logits, t_logits):
                return F.mse_loss(s_logits, t_logits.detach())
        else:
            raise ValueError(f"Invalid consistency_loss_type: {self.consistency_loss_type}")

        consistency_loss = loss_fn(student_logits_x0, teacher_logits_x0)

        loss = ce_loss + self._consistency_weight * consistency_loss

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

            self.val_confmat_student.update(student_probs, y)
            self.val_confmat_teacher.update(teacher_probs, y)

            val_nll = F.nll_loss(teacher_probs.clamp_min(1e-8).log(), y)
            self.log("val/logloss", val_nll, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")

        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]

        if hasattr(opt, "param_groups") and len(opt.param_groups) > 0:
            lr = opt.param_groups[0].get("lr", None)
            if lr is not None:
                self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc_student", self.val_acc_student.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc_teacher", self.val_acc_teacher.compute(), on_epoch=True, prog_bar=True, sync_dist=True)

        cm_student = self.val_confmat_student.compute()
        cm_teacher = self.val_confmat_teacher.compute()

        self._final_confmat_student = cm_student.detach().cpu()
        self._final_confmat_teacher = cm_teacher.detach().cpu()

        self.val_acc_student.reset()
        self.val_acc_teacher.reset()
        self.val_confmat_student.reset()
        self.val_confmat_teacher.reset()

    def on_fit_end(self):
        if self.trainer is None or not self.trainer.is_global_zero:
            return

        save_dir = "output/confusion_matrix"

        if self._final_confmat_student is not None:
            save_confusion_matrix(
                self._final_confmat_student,
                save_path=f"{save_dir}/student.png",
                title="Confusion Matrix (Student)"
            )
            self.print(f"Saved: {save_dir}/student.png")

        if self._final_confmat_teacher is not None:
            save_confusion_matrix(
                self._final_confmat_teacher,
                save_path=f"{save_dir}/teacher.png",
                title="Confusion Matrix (Teacher)"
            )
            self.print(f"Saved: {save_dir}/teacher.png")

    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        depth = batch.get("depth") if self._uses_depth else None
        x = self._IMGNET_NORMALIZE(x)

        logits = self.teacher_forward(x, depth)
        logits = self._maybe_clip_logits(logits)
        probs = F.softmax(logits, dim=1)
        return {"preds": probs, "img_name": batch["img_name"]}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler == "none":
            return opt

        if self.scheduler == "onecycle":
            if self.trainer is None:
                raise RuntimeError("Trainer not attached")

            total_steps = int(self.trainer.estimated_stepping_batches)
            self.total_steps = total_steps
            self.consistency_rampup_steps = int(self.consistency_rampup_steps * total_steps)

            sched: LRScheduler = OneCycleLR(
                opt,
                max_lr=self.lr,
                total_steps=max(1, total_steps),
                pct_start=0.1,
                div_factor=10.0,
                final_div_factor=100.0,
                anneal_strategy="cos",
            )

            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        return opt
