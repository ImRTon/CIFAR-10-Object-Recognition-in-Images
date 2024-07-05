# CIFAR-10 - Object Recognition in Images
A simple image classification project using PyTorch Lightning with Deep Learning.

## Training
Arguments:
```bash
python main.py fit --data.batch_size 256 --data.data_dir=datasets/train --data.label_path=datasets/trainLabels.csv --model.backbone ResNet50 --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=step --trainer.max_epochs 100 --model.lr 6e-2 --model.weight_decay 1e-4 --model.momentum 0.9
```

* `data.data_dir` - Path to the directory containing training images
* `data.label_path` - Path to the CSV file containing image labels
* `model.backbone` - Backbone model to use for training. Available options: `ResNet18`, `ResNet50`, `Simple`  
* `model.weight_decay` - Weight decay for SGD optimizer  
* `model.momentum` - Momentum for SGD optimizer  

## Predicting
Arguments:
```bash
python predict.py -d "TEST_IMGS_PATH" -l "TEST_LABEL_PATH" -f "CONFIG_PATH" -c "CHECKPOINT_PATH"
```

* `d` - Path to the directory containing test images
* `l` - Path to the CSV file containing image labels, if not exists, the file will be created
* `f` - Path to the configuration file
* `c` - Path to the checkpoint file

## Results
| Method              | Optimizer | Scheduler | Validation Accuracy  | Validation Accuracy  | Version |
|---------------------|-----------|-----------|----------------------|----------------------|---------|
| Simple Residual     | SGD       | OneCycleLR| 0.835                | 0.8247               | 13      |
| Simple SiLU Residual| SGD       | OneCycleLR| 0.825                | -                    | 21      |
| ResNet18            | SGD       | OneCycleLR| 0.927                | 0.9269               | 12      |
| ResNet50            | SGD       | OneCycleLR| 0.953                | 0.9540               | 20      |
