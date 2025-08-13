import torch
import torch.nn as nn
from arguments import parse_arguments, print_args
from src.models.model import UNet
from src.trainer.main_trainer import MainTrainer
from src.data.loader import get_dataloader
from config.config import get_config


def main():
    args = parse_arguments()
    print_args(args)
    
    # Get dataset configuration
    config = get_config(args.dataset)
    args.num_classes = config['num_classes']
    
    # Initialize model
    if args.model == 'unet':
        model = UNet(
            channels=[32, 64, 128, 256], 
            num_res_blocks=2,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloader(args)
    
    # Initialize trainer
    trainer = MainTrainer(model, args, train_loader, val_loader, test_loader)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()