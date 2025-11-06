import lightning as L
import torch
import torch.nn.functional as F
from src.DriverNet.models.backbone.vit import vit_model_names, load_vit

class Teacher(L.LightningModule):
    def __init__(self, name: vit_model_names = "vit_b_16", num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.model = load_vit(name, num_classes, pretrained)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        y_hat = self(x)
        return F.softmax(y_hat, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
