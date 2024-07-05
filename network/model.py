from typing import Any
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import models
from torchmetrics import Accuracy
from pathlib import Path

import lightning as L
import json
import torch

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU(inplace=True),
    "leakyrelu": nn.LeakyReLU(inplace=True),
    "silu": nn.SiLU(inplace=True),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
}

def get_activation(act_func: str) -> nn.Module:
    act_func = act_func.lower()
    if act_func in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_func]
    else:
        raise ValueError(f"Activation function {act_func} is not implemented.")

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1,
        act_func: str = "ReLU"
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=1, 
            bias=False # BatchNorm2d has bias
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.act1 = get_activation(act_func)
        self.conv2 = nn.Conv2d(
            output_channels, 
            output_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=1, 
            bias=False # BatchNorm2d has bias
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.act2 = get_activation(act_func)

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out
    
class SimpleResidualNet(nn.Module):
    def __init__(
        self, 
        num_class: int,
        act_func: str = "GeLU",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.act = get_activation(act_func)
        self.layer = nn.Sequential(
            ResidualBlock(32, 64, 3, stride=1, act_func=act_func),
            ResidualBlock(64, 128, 3, stride=2, act_func=act_func),
            ResidualBlock(128, 256, 3, stride=2, act_func=act_func),
            ResidualBlock(256, 512, 3, stride=2, act_func=act_func),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_class),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.layer(x)

        return x
    
class ResNet18(nn.Module):
    def __init__(self, num_class: int) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_class)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet(x)
    
class ResNet50(nn.Module):
    def __init__(self, num_class: int) -> None:
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_class)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet(x)

class SimmpleCNN(L.LightningModule):
    def __init__(
        self, 
        lr: float, 
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        backbone: str = "resnet18",
        num_classes: int | None = None,
        class_weights: list[float] | None = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["num_classes", "class_weights"])

        if self.hparams.backbone.lower() == "resnet18":
            self.layer = ResNet18(num_classes)
        elif self.hparams.backbone.lower() == "resnet50":
            self.layer = ResNet50(num_classes)
        elif self.hparams.backbone.lower() == "simple":
            self.layer = SimpleResidualNet(num_classes)
        else:
            raise NotImplementedError(f"Backbone {backbone} is not implemented.")
        
        if class_weights is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            cls_weights = torch.tensor(class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=cls_weights)

        self.accu_train = Accuracy(task="multiclass", num_classes=num_classes)
        self.accu_val = Accuracy(task="multiclass", num_classes=num_classes)
        self.accu_test = Accuracy(task="multiclass", num_classes=num_classes)

    def setup_weight(self, weights: Tensor):
        self.criterion.weight = weights

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
    
    def _shared_step(self, batch, batch_idx):
        data, target = batch
        pred = self(data)
        loss = self.criterion(pred, target)
        pred_labels = pred.argmax(dim=1)
        return loss, pred_labels, target

    def training_step(self, batch, batch_idx):
        loss, pred_labels, target = self._shared_step(batch, batch_idx)
        accu = self.accu_train(pred_labels, target)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accu, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, pred_labels, target = self._shared_step(batch, batch_idx)
        accu = self.accu_val(pred_labels, target)
        self.log_dict({"val_loss": loss, "val_acc": accu}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, pred_labels, target = self._shared_step(batch, batch_idx)
        accu = self.accu_test(pred_labels, target)
        self.log_dict({"test_loss": loss, "test_acc": accu})

    def predict_step(self, batch, batch_idx):
        data, id = batch
        pred = self(data)
        pred_labels = pred.argmax(dim=1)

        return pred_labels, id

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), 
            self.hparams.lr, 
            momentum=self.hparams.momentum, 
            weight_decay=self.hparams.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=self.hparams.lr, 
                    total_steps=self.trainer.estimated_stepping_batches,
                ),
                "interval": "step",
            }
            
        }