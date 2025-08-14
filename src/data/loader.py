from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import CustomDataset
from .utils import collect_image_paths
from .collate import collate_fn


def get_transforms(args, is_train=True):
    """Create transforms based on args configuration"""
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize((args.resize, args.resize)))
    
    # Training augmentations
    if is_train:
        if args.random_crop:
            transform_list.append(transforms.RandomCrop(args.crop_size, padding=4))
        
        if args.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # Auto augmentation
        if args.auto_augment != 'none':
            if args.auto_augment == 'randaugment':
                transform_list.append(transforms.RandAugment())
            elif args.auto_augment == 'autoaugment':
                transform_list.append(transforms.AutoAugment())
            elif args.auto_augment == 'trivialaugment':
                transform_list.append(transforms.TrivialAugmentWide())
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    if args.normalize:
        if hasattr(args, 'mean') and hasattr(args, 'std'):
            transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
        else:
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    return transforms.Compose(transform_list)


def create_dataloader(
    data_dir: Path,
    args,
    batch_size: int,
    shuffle: bool,
    is_train: bool = False,
    return_class_mapping: bool = False
    ) -> DataLoader:
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths, class_mapping = collect_image_paths(data_dir, extensions=extensions)
    
    transform = get_transforms(args, is_train=is_train)
    dataset = CustomDataset(image_paths, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn
    )
    
    if return_class_mapping:
        return dataloader, class_mapping
    return dataloader


def get_dataloader(args):
    train_loader = create_dataloader(
        args.data_path + '/train',
        args,
        args.batch_size,
        shuffle=True,
        is_train=True
    )
    
    val_loader = create_dataloader(
        args.data_path + '/val',
        args, 
        args.val_batch_size,
        shuffle=False,
        is_train=False
    )
    
    test_loader = create_dataloader(
        args.data_path + '/test',
        args,
        args.test_batch_size, 
        shuffle=False,
        is_train=False
    )
    
    return train_loader, val_loader, test_loader