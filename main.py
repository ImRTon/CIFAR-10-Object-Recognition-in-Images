from lightning.pytorch.cli import LightningCLI

from network.model import SimmpleCNN
from network.data import CIFAR10DataModule

class CIFAR10CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.class_weights", "model.class_weights", apply_on="instantiate")

def main():
    cli = CIFAR10CLI(
        SimmpleCNN, 
        CIFAR10DataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == '__main__':
    main()