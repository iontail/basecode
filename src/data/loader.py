from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch
import numpy as np
from collections import Counter

from .dataset import CustomDataset, BaseDataset, MemoryEfficientDataset
from .utils import collect_image_paths, create_balanced_split
from .collate import collate_fn, mixup_collate_fn


def get_transforms(args, is_train=True):
    """Create transforms based on args configuration"""
    transform_list = []
    
    # Resize
    if hasattr(args, 'resize'):
        transform_list.append(transforms.Resize((args.resize, args.resize)))
    
    # Training augmentations
    if is_train and hasattr(args, 'augmentation') and args.augmentation:
        if hasattr(args, 'random_crop') and args.random_crop:
            crop_size = getattr(args, 'crop_size', 224)
            transform_list.append(transforms.RandomCrop(crop_size, padding=4))
        
        if hasattr(args, 'random_flip') and args.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # Color jittering
        if hasattr(args, 'color_jitter') and args.color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))
        
        # Random rotation
        if hasattr(args, 'random_rotation') and args.random_rotation:
            transform_list.append(transforms.RandomRotation(10))
        
        # Auto augmentation
        if hasattr(args, 'auto_augment') and args.auto_augment != 'none':
            if args.auto_augment == 'randaugment':
                transform_list.append(transforms.RandAugment())
            elif args.auto_augment == 'autoaugment':
                transform_list.append(transforms.AutoAugment())
            elif args.auto_augment == 'trivialaugment':
                transform_list.append(transforms.TrivialAugmentWide())
        
        # Gaussian blur
        if hasattr(args, 'gaussian_blur') and args.gaussian_blur:
            transform_list.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    if hasattr(args, 'normalize') and args.normalize:
        if hasattr(args, 'mean') and hasattr(args, 'std'):
            transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
        else:
            # Default ImageNet normalization
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
    
    return transforms.Compose(transform_list)


def create_balanced_sampler(labels: list, num_samples: int = None):
    """Create a balanced sampler for handling class imbalance"""
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[label] for label in labels]
    
    if num_samples is None:
        num_samples = len(weights)
    
    return WeightedRandomSampler(weights, num_samples, replacement=True)


def create_dataloader(
    data_dir: Path,
    args,
    batch_size: int,
    shuffle: bool,
    is_train: bool = False,
    return_class_mapping: bool = False,
    use_balanced_sampling: bool = False
    ) -> DataLoader:
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths, class_mapping = collect_image_paths(data_dir, extensions=extensions)
    
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    
    transform = get_transforms(args, is_train=is_train)
    
    # Choose dataset type based on memory efficiency setting
    dataset_class = MemoryEfficientDataset if getattr(args, 'memory_efficient', False) else CustomDataset
    dataset = dataset_class(image_paths, transform=transform)
    
    # Create sampler for balanced sampling if requested
    sampler = None
    if use_balanced_sampling and is_train:
        labels = [label for _, label in image_paths]
        sampler = create_balanced_sampler(labels)
        shuffle = False  # Can't shuffle when using sampler
    
    # Choose collate function based on augmentation settings
    collate_function = collate_fn
    if is_train and hasattr(args, 'mixup') and args.mixup:
        collate_function = mixup_collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=getattr(args, 'pin_memory', True),
        drop_last=is_train,  # Drop last incomplete batch during training
        collate_fn=collate_function,
        persistent_workers=getattr(args, 'num_workers', 4) > 0,
        prefetch_factor=2 if getattr(args, 'num_workers', 4) > 0 else 2
    )
    
    # Print dataset statistics
    if is_train:
        class_dist = dataset.get_class_distribution()
        print(f"Dataset statistics for {data_dir.name}:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Number of classes: {len(class_mapping)}")
        print(f"  Class distribution: {dict(sorted(class_dist.items()))}")
    
    if return_class_mapping:
        return dataloader, class_mapping
    return dataloader


def get_dataloader(args, return_class_mapping=False):
    """Create train, validation, and test dataloaders"""
    data_path = Path(args.data_path)
    
    # Check if split directories exist, otherwise create split
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'  
    test_dir = data_path / 'test'
    
    loaders = {}
    class_mapping = None
    
    # Create train loader if directory exists
    if train_dir.exists():
        train_loader, class_mapping = create_dataloader(
            train_dir,
            args,
            args.batch_size,
            shuffle=True,
            is_train=True,
            return_class_mapping=True,
            use_balanced_sampling=getattr(args, 'balanced_sampling', False)
        )
        loaders['train'] = train_loader
    
    # Create validation loader if directory exists
    if val_dir.exists():
        val_loader = create_dataloader(
            val_dir,
            args, 
            getattr(args, 'val_batch_size', args.batch_size),
            shuffle=False,
            is_train=False
        )
        loaders['val'] = val_loader
    
    # Create test loader if directory exists
    if test_dir.exists():
        test_loader = create_dataloader(
            test_dir,
            args,
            getattr(args, 'test_batch_size', args.batch_size), 
            shuffle=False,
            is_train=False
        )
        loaders['test'] = test_loader
    
    # If no split directories exist, create automatic split
    if not any([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
        print("No train/val/test directories found. Creating automatic split...")
        train_loader, val_loader, test_loader, class_mapping = create_split_dataloaders(args)
        loaders.update({'train': train_loader, 'val': val_loader, 'test': test_loader})
    
    # Return based on what directories exist
    result = []
    for split in ['train', 'val', 'test']:
        if split in loaders:
            result.append(loaders[split])
        else:
            result.append(None)
    
    if return_class_mapping:
        result.append(class_mapping)
    
    return tuple(result)


def create_split_dataloaders(args):
    """Create dataloaders with automatic train/val/test split"""
    data_path = Path(args.data_path)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths, class_mapping = collect_image_paths(data_path, extensions=extensions)
    
    # Create balanced split
    split_ratio = getattr(args, 'split_ratio', [0.8, 0.1, 0.1])
    train_paths, val_paths, test_paths = create_balanced_split(image_paths, split_ratio)
    
    # Create datasets
    train_transform = get_transforms(args, is_train=True)
    val_transform = get_transforms(args, is_train=False)
    
    dataset_class = MemoryEfficientDataset if getattr(args, 'memory_efficient', False) else CustomDataset
    
    train_dataset = dataset_class(train_paths, transform=train_transform)
    val_dataset = dataset_class(val_paths, transform=val_transform)
    test_dataset = dataset_class(test_paths, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=getattr(args, 'pin_memory', True),
        drop_last=True,
        persistent_workers=getattr(args, 'num_workers', 4) > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(args, 'val_batch_size', args.batch_size),
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=getattr(args, 'pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=getattr(args, 'test_batch_size', args.batch_size),
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 4),
        pin_memory=getattr(args, 'pin_memory', True)
    )
    
    print(f"Created automatic split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples") 
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, class_mapping