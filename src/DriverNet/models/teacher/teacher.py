import torch
import torch.nn.functional as F

from src.DriverNet.models.base import BaseModel


class Teacher(BaseModel):
    def test_step(self, batch, batch_idx):
        x0 = batch["pixel_values"]
        x1 = batch.get("pixel_values_proc")
        m = batch.get("has_proc")
        y = batch["labels"].long()

        logit0 = self._maybe_clip_logits(self(x0))
        p0 = F.softmax(logit0, dim=-1)
        ce0 = self.ce(logit0, y)

        if m is None or not bool(m.any()):
            probs_avg = p0
            ce = ce0
        else:
            logit1 = self._maybe_clip_logits(self(x1))
            p1 = F.softmax(logit1, dim=-1)
            ce1 = self.ce(logit1[m], y[m]) if bool(m.any()) else torch.zeros_like(ce0)
            probs_avg = 0.5 * (p0 + p1)
            ce = ce0 + ce1

        self.test_acc.update(probs_avg, y)
        self.log("test/logloss", ce, on_epoch=True, prog_bar=True, sync_dist=True)
        return ce

    def on_test_epoch_end(self):
        self.log("test/acc", self.test_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_acc.reset()

    def predict_step(self, batch, batch_idx):
        x0 = batch["pixel_values"]
        x1 = batch.get("pixel_values_proc")
        m = batch.get("has_proc")
        img_names = batch["img_name"]

        logit0 = self._maybe_clip_logits(self(x0))
        p0 = F.softmax(logit0, dim=1)

        if m is None or not bool(m.any()):
            probs = p0
        else:
            logit1 = self._maybe_clip_logits(self(x1))
            p1 = F.softmax(logit1, dim=1)
            probs = 0.5 * (p0 + p1)

        return {"preds": probs, "img_name": img_names}
