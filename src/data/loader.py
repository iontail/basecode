from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import ImageDataset
from .mapping import get_label_map, get_image_label_list
from .collator import default_collate_fn


def create_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224,
    collate_fn = default_collate_fn
) -> DataLoader:

    label_map = get_label_map(data_dir)
    image_label_list = get_image_label_list(data_dir, label_map)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = ImageDataset(image_label_list, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn  # 여기 주의
    )
