from pathlib import Path
from typing import Callable, List, Tuple, Dict, Optional, Union
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import logging

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(Dataset):
    """Base dataset class with common functionality"""
    
    def __init__(
        self, 
        image_paths: List[Tuple[Path, int]], 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_images: bool = False
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        
        # Validate all paths exist
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that all image paths exist"""
        invalid_paths = []
        for img_path, _ in self.image_paths:
            if not img_path.exists():
                invalid_paths.append(img_path)
        
        if invalid_paths:
            logging.warning(f"Found {len(invalid_paths)} invalid paths")
            self.image_paths = [(p, l) for p, l in self.image_paths if p.exists()]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path, label = self.image_paths[idx]
        
        # Load from cache or file
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            try:
                image = Image.open(img_path).convert("RGB")
                if self.cache_images:
                    self.image_cache[idx] = image
            except Exception as e:
                logging.error(f"Error loading image {img_path}: {e}")
                # Return a dummy black image in case of error
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        original_image = image.copy() if hasattr(image, 'copy') else image
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return {
            "image": image,
            "label": label,
            "path": str(img_path),
            "original_image": original_image
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in dataset"""
        class_counts = {}
        for _, label in self.image_paths:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    
    def get_image_paths_by_class(self, class_idx: int) -> List[Path]:
        """Get all image paths for a specific class"""
        return [path for path, label in self.image_paths if label == class_idx]


class CustomDataset(BaseDataset):
    """Custom dataset for image classification"""
    pass


class MemoryEfficientDataset(BaseDataset):
    """Memory efficient dataset that loads images on-the-fly"""
    
    def __init__(self, *args, **kwargs):
        # Force cache_images to False for memory efficiency
        kwargs['cache_images'] = False
        super().__init__(*args, **kwargs)