from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
import json
import lightning as L
import pandas as pd
import numpy as np

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_dir: Path | str, 
        label_path: Path | str, 
        split_ratio: float=0.85, 
        batch_size: int = 64,
        target_size: tuple[int, int] | int = 32,
        data_config_path: Path | str | None = None,
        num_classes: int | None = None,
        class_weights: list[float] | None = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.label_path = Path(label_path)
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.target_size = target_size

        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)

        self.data_config = {
            "norm_dict": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "class_mapping": None,
            "class_weights": None,
        }

        if data_config_path is None:
            data_config_path = self.data_dir / "data_config.json"

        if data_config_path.exists():
            with open(data_config_path, "r") as f:
                self.data_config = json.load(f)
        self.data_config_path = data_config_path

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found.")
        if not self.label_path.exists():
            raise FileNotFoundError(f"Label file {self.label_path} not found.")
        
        self.df_data = pd.read_csv(self.label_path)

        if self.data_config["class_mapping"] is None:
            self.data_config["class_mapping"] = {key: i for i, key in enumerate(self.df_data["label"].unique())}

        if num_classes is None:
            num_classes = len(self.data_config["class_mapping"])
        self.num_classes = num_classes

        self.df_data["label"] = self.df_data["label"].map(self.data_config["class_mapping"])
        if "class_weights" not in self.data_config or self.data_config["class_weights"] is None:
            class_weights = self.df_data["label"].value_counts(normalize=True).sort_index()
            class_weights = num_classes * class_weights
            class_weights = class_weights.to_list()
            class_weights = [max(0.1, min(class_weight, 10)) for class_weight in class_weights]
            self.data_config["class_weights"] = class_weights
            self.class_weights = class_weights
        else:
            self.class_weights = self.data_config["class_weights"]
        self.dump_config()
        
        self.save_hyperparameters()

    def dump_config(self):
        with open(self.data_config_path, "w") as f:
            print("Dumping config to", self.data_config_path)
            json.dump(self.data_config, f)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            labels = self.df_data["label"].to_list()
            paths = self.df_data["id"].apply(lambda x: self.data_dir / f"{x}.png")
            paths = paths.to_list()

            self.train_paths, self.val_paths, self.train_labels, self.val_labels = train_test_split(
                paths, labels, train_size=self.split_ratio)

        elif stage == "test":
            labels = self.df_data["label"].to_list()
            paths = self.df_data["id"].apply(lambda x: self.data_dir / f"{x}.png")
            paths = paths.to_list()
        elif stage == "predict":
            paths = self.df_data["id"].apply(lambda x: self.data_dir / f"{x}.png")
            paths = paths.to_list()

    def train_dataloader(self):
        return DataLoader(
            CIFARDataset(
                self.train_paths, 
                self.data_config["norm_dict"],
                self.target_size,
                is_train=True,
                labels=self.train_labels
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=20
        )
    
    def val_dataloader(self):
        return DataLoader(
            CIFARDataset(
                self.val_paths, 
                self.data_config["norm_dict"],
                self.target_size,
                is_train=False,
                labels=self.val_labels
            ),
            batch_size=self.batch_size,
            num_workers=20
        )
    
    def test_dataloader(self):
        return DataLoader(
            CIFARDataset(
                self.test_paths, 
                self.data_config["norm_dict"],
                self.target_size,
                is_train=False,
                labels=self.test_labels
            ),
            batch_size=self.batch_size,
            num_workers=20
        )
    
    def predict_dataloader(self):
        return DataLoader(
            CIFARDataset(
                self.predict_paths, 
                self.data_config["norm_dict"],
                self.target_size,
                is_train=False
            ),
            batch_size=self.batch_size,
            num_workers=20
        )

class CIFARDataset(Dataset):
    def __init__(
        self, 
        data_paths: list[Path],  
        norm_dict: dict, 
        target_size: tuple[int, int] | int,
        is_train: bool,
        labels: list[int] | None = None,
    ):
        self.data_paths = data_paths
        self.labels = labels
        self.norm_dict = norm_dict

        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        if target_size[0] < 0 or target_size[1] < 0:
            raise ValueError("Target size must be positive integers.")

        if labels is not None and len(self.data_paths) != len(self.labels):
            raise ValueError("Datas and labels must have the same length.")

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.Resize((target_size[0], target_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(self.norm_dict["mean"], self.norm_dict["std"])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((target_size[0], target_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(self.norm_dict["mean"], self.norm_dict["std"])
            ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image = Image.open(self.data_paths[idx]).convert("RGB")
        if self.labels is None:
            return self.transform(image), self.data_paths[idx]
        else:
            return self.transform(image), torch.tensor(self.labels[idx])