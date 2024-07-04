# CIFAR-10 - Object Recognition in Images
a simple image classification project using PyTorch Lightning with Deep Learning.

## Training
Arguments:
```bash
python main.py fit --data.batch_size 256 --data.data_dir=datasets/train --data.label_path=datasets/trainLabels.csv --model.backbone ResNet50 --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=step --trainer.max_epochs 100 --model.lr 6e-2 --model.weight_decay 1e-4 --model.momentum 0.9
```

## Predicting
Arguments:
```bash
python predict.py -d "TEST_IMGS_PATH" -l "TEST_LABEL_PATH" -f "CONFIG_PATH" -c "CHECKPOINT_PATH"
```