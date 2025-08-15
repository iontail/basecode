import argparse
import os
from datetime import datetime

def parse_arguments():
    """
    Parse command line arguments for training configuration
    Returns:
        - args: parsed arguments namespace with all training parameters
    """
    parser = argparse.ArgumentParser(description='Image Classification Training Configuration')

    # Reproducibility arguments
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic algorithms for reproducibility')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/dataset',
                        help='Path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Name of the dataset to use', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Ratio for train, validation, and test split')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Use pinned memory for data loading')
    
    # Data augmentation arguments
    parser.add_argument('--random_crop', action='store_true', default=True,
                        help='Apply random cropping to images')
    parser.add_argument('--random_flip', action='store_true', default=True,
                        help='Apply random horizontal flip to images')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize the dataset')
    parser.add_argument('--resize', type=int, default=224,
                        help='Resize the input images to this size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size for training images')
    parser.add_argument('--augmentation', action='store_true', default=False,
                        help='Apply data augmentation techniques')
    parser.add_argument('--auto_augment', type=str, default='none',
                        choices=['none', 'randaugment', 'autoaugment', 'trivialaugment'],
                        help='Auto augmentation policy')
    parser.add_argument('--cutmix', action='store_true', default=False,
                        help='Use CutMix data augmentation')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='Use Mixup data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Alpha parameter for Mixup')
    parser.add_argument('--color_jitter', action='store_true', default=False,
                        help='Apply color jittering')
    parser.add_argument('--random_rotation', action='store_true', default=False,
                        help='Apply random rotation')
    parser.add_argument('--gaussian_blur', action='store_true', default=False,
                        help='Apply gaussian blur')
    parser.add_argument('--balanced_sampling', action='store_true', default=False,
                        help='Use balanced sampling for imbalanced datasets')
    parser.add_argument('--memory_efficient', action='store_true', default=False,
                        help='Use memory efficient dataset loading')
    parser.add_argument('--return_original', action='store_true', default=False,
                        help='Return original image along with transformed image')

    # Model architecture arguments
    parser.add_argument('--model', type=str, default='unet',
                        help='Model to use', choices=['unet', 'resnet18', 'resnet50', 'vit'])
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size for the dataset')
    parser.add_argument('--mean', type=float, nargs=3, default=[0.4914, 0.4822, 0.4465],
                        help='Mean values for normalization')
    parser.add_argument('--std', type=float, nargs=3, default=[0.2023, 0.1994, 0.2010],
                        help='Standard deviation values for normalization')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for the model')
    parser.add_argument('--expansion_ratio', type=float, default=4.0,
                        help='Ratio of expansion to hidden size')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Use pretrained model')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer to use', choices=['Adam', 'AdamW', 'SGD', 'RMSprop'])
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for the optimizer for SGD')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for the optimizer (L2 Regularization)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='Learning rate scheduler to use', 
                        choices=['step', 'cosine', 'plateau', 'exponential', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs for the learning rate scheduler')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6,
                        help='Starting learning rate for warmup phase')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma value for step scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.1,
                        help='Factor for ReduceLROnPlateau schedulerz')
    
    # Loss function arguments
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor')

    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Frequency of evaluation during training (in epochs)')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Only evaluate model without training')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Only run test evaluation')

    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--load_best', action='store_true', default=False,
                        help='Load best model instead of latest when resuming')

    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save the training logs')
    parser.add_argument('--experiment_name', type=str, 
                        default='',
                        help='Name of the experiment for logging')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='my_project',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity (team) name')
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use TensorBoard for logging')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='Print frequency during training')

    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for training')
    parser.add_argument('--use_multigpu', action='store_true', default=False,
                        help='Use multiple GPUs for training (DataParallel)')
    parser.add_argument('--use_ddp', action='store_true', default=False,
                        help='Use DistributedDataParallel for multi-GPU training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training (AMP)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile for optimization (PyTorch 2.0+)')
    
    # Model saving arguments
    parser.add_argument('--save_best', action='store_true', default=True,
                        help='Save the best model based on validation performance')
    parser.add_argument('--save_last', action='store_true', default=True,
                        help='Save the last model checkpoint')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Frequency of saving the model (in epochs)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save the model checkpoints')
    parser.add_argument('--keep_checkpoint_max', type=int, default=5,
                        help='Maximum number of checkpoints to keep')

    # Debug and testing arguments
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode (use small subset of data)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Do not save anything, just test the pipeline')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Profile the training process')
    parser.add_argument('--fast_dev_run', action='store_true', default=False,
                        help='Run one batch for debugging')

    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='Minimum delta for early stopping')

    args = parser.parse_args()

    if args.device == 'auto':
        import torch
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.grad_clip < 0:
        raise ValueError("Gradient clipping value must be non-negative")
    
    if sum(args.split_ratio) != 1.0:
        print(f"Warning: Split ratio sum is {sum(args.split_ratio)}, normalizing to 1.0")
        total = sum(args.split_ratio)
        args.split_ratio = [x/total for x in args.split_ratio]
    
    if not args.experiment_name or args.experiment_name == '':
        print("Experiment name not provided, generating a timestamped name.")
        args.experiment_name = f'{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    
    return args

def print_args(args):
    """
    Print training configuration in organized categories
    Args:
        - args: parsed arguments namespace to display
    """
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    
    categories = {
        'Experiment': ['experiment_name', 'seed', 'device', 'debug'],
        'Data': ['dataset', 'data_path', 'batch_size', 'num_workers'],
        'Model': ['model', 'num_classes', 'hidden_size', 'pretrained'],
        'Training': ['epochs', 'lr', 'optimizer', 'scheduler', 'weight_decay'],
        'Logging': ['use_wandb', 'use_tensorboard', 'save_dir', 'log_dir']
    }
    
    for category, keys in categories.items():
        print(f"\n[{category}]")
        for key in keys:
            if hasattr(args, key):
                value = getattr(args, key)
                print(f"  {key:<20}: {value}")
        print("-" * 40)
    

if __name__ == "__main__":
    args = parse_arguments()
    print_args(args)