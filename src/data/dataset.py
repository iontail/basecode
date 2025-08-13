from pathlib import Path
from typing import Callable, List, Tuple, Dict, Optional
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, 
        image_paths: List[Tuple[Path, int]], 
        transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path, label = self.image_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": img_path
        }