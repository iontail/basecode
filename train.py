import torch
import torch.nn as nn
from arguments import parse_arguments, print_args
from src.models.model import UNet
from src.trainer.main_trainer import MainTrainer
from src.data.loader import get_dataloader
def main():
    """
    Main training function
    Parses arguments, creates model and trainer, then starts training
    """
    args = parse_arguments()
    print_args(args)
    
    if args.model == 'unet':
        model = UNet(
            channels=[32, 64, 128, 256], 
            num_res_blocks=2,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    train_loader, val_loader, test_loader = get_dataloader(args, extensions=args.extensions)
    trainer = MainTrainer(model, args, train_loader, val_loader, test_loader)
    trainer.train()


if __name__ == "__main__":
    main()