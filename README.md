# CIFAR-10 - Object Recognition in Images
a simple image classification project using PyTorch Lightning with Deep Learning.

## Training
Arguments:
```bash
python main.py fit --data.data_path=datasets/train.csv --model.output_dim 2 --model.input_dim 9 --model.lr 2e-3 --trainer.callbacks+=EarlyStopping --trainer.callbacks.monitor val_loss --trainer.callbacks.patience=10 --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=epoch --trainer.max_epochs 100
```

## Predicting
Arguments:
```bash
python predict.py -d "TEST_DATA_PATH" -c "CHECKPOINT_PATH"
```