from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import ImageDataset
from .utils import collect_image_paths
from .collate import collate_fn


def create_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224,
    normalize: bool = True
) -> DataLoader:
    """Create DataLoader for image classification."""
    
    # Collect all image paths with labels
    image_paths = collect_image_paths(data_dir)
    
    # ============================
    # Define transforms
    # or fill custom transformations in dataset.py
    # ============================
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    transform = transforms.Compose(transform_list)
    
    # Create dataset
    dataset = ImageDataset(image_paths, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )