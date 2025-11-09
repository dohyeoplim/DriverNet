import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from src.DriverNet.models.backbone.vit import vit_model_names, load_vit

class Teacher(L.LightningModule):
    def __init__(self, name: vit_model_names = "vit_b_16", num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.model = load_vit(name, num_classes, pretrained)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        img_names = batch["img_name"]
        y_hat = self(x)

        return {"preds": F.softmax(y_hat, dim=1), "img_name": img_names}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
