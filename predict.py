import argparse
import csv
import torch
import lightning as L
import json

from pathlib import Path

from network.model import SimmpleCNN
from network.data import CIFAR10DataModule

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-l", "--label_path", type=str, required=True)
    parser.add_argument("-f", "--config_path", type=str, required=True)
    parser.add_argument("-c", "--checkpoint_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, default="predictions.csv")
    return parser.parse_args()

def main():
    args = get_args()

    label_path = Path(args.label_path)
    if not label_path.exists():
        print("Label file not found. Generating labels from data directory.")
        with open(args.label_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            for img_path in Path(args.data_path).glob("*.png"):
                id = int(img_path.stem)
                writer.writerow([id, "unknown"])
                
    config = {}
    with open(args.config_path, "r") as f:
        config = json.load(f)

    model = SimmpleCNN.load_from_checkpoint(
        args.checkpoint_path, 
        num_classes=len(config["class_mapping"]), 
        class_weights=config["class_weights"]
    )
    data_module = CIFAR10DataModule.load_from_checkpoint(
        args.checkpoint_path, 
        label_path=args.label_path, 
        data_dir=args.data_path, 
        num_classes=len(config["class_mapping"])
    )

    trainer = L.Trainer()

    outputs = trainer.predict(model, data_module)

    results = []
    ids = []
    for batch in outputs:
        result, id = batch
        results.extend(result.cpu().numpy())
        ids.extend(id.cpu().numpy())

    if config["class_mapping"] is None:
        raise ValueError("class_mapping is not initialized")
    mapping = {idx: key for key, idx in config["class_mapping"].items()}

    with open(args.output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for result, id in zip(results, ids):
            writer.writerow([id, mapping[result]])


if __name__ == '__main__':
    main()