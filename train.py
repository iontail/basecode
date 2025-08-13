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
    
    config = get_config(args.dataset)
    args.num_classes = config['num_classes']
    
    if args.model == 'unet':
        model = UNet(
            channels=[32, 64, 128, 256], 
            num_res_blocks=2,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    train_loader, val_loader, test_loader = get_dataloader(args)
    trainer = MainTrainer(model, args, train_loader, val_loader, test_loader)
    trainer.train()


if __name__ == "__main__":
    main()