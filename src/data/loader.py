from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import CustomDataset
from .utils import collect_image_paths
from .collate import collate_fn


def create_dataloader(
    data_dir: Path,
    batch_size: int,
    shuffle: bool,
    image_size: int = 224,
    num_workers: int = 4,
    normalize: bool = True,
    return_class_mapping: bool = False
    ) -> DataLoader:
    
    extensions = ['.jpg']
    image_paths, class_mapping = collect_image_paths(data_dir, extensions=extensions)
    
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    transform = transforms.Compose(transform_list)
    dataset = CustomDataset(image_paths, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) if not return_class_mapping else (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        ),
        class_mapping
    )


def get_dataloader(args):
    train_loader = create_dataloader(
        args.data_path + '/train',
        args.batch_size,
        shuffle=True,
        image_size=args.resize,
        num_workers=args.num_workers
    )
    
    val_loader = create_dataloader(
        args.data_path + '/val', 
        args.val_batch_size,
        shuffle=False,
        image_size=args.resize,
        num_workers=args.num_workers
    )
    
    test_loader = create_dataloader(
        args.data_path + '/test',
        args.test_batch_size, 
        shuffle=False,
        image_size=args.resize,
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader, test_loader