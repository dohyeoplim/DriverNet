import lightning as L
import torch.nn.functional as F
from src.DriverNet.models.backbone.vit import vit_model_names, load_vit

class Teacher(L.LightningModule):
    def __init__(self, model: vit_model_names = "vit_b_16", num_classes: int = 10):
        super().__init__()
        self.model = load_vit(model, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss
