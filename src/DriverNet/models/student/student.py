import torch.nn.functional as F

from src.DriverNet.models.base import BaseModel

class Student(BaseModel):
    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        y_hat = self._maybe_clip_logits(self(x))
        probs = F.softmax(y_hat, dim=1)
        return {"preds": probs, "img_name": batch["img_name"]}
