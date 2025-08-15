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
        cache_images: bool = False,
        return_original: bool = False
    ):
        """
        Initialize BaseDataset
        Args:
            - image_paths: list of tuples (image_path, label)
            - transform: optional transform to apply to images
            - target_transform: optional transform to apply to labels
            - cache_images: whether to cache loaded images in memory
            - return_original: whether to return original image along with transformed image
        """
        self.image_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.return_original = return_original
        self.image_cache = {} if cache_images else None
        
        # Validate all paths exist
        self._validate_paths()
    
    def _validate_paths(self):
        """
        Validate that all image paths exist and remove invalid ones
        """
        invalid_paths = []
        for img_path, _ in self.image_paths:
            if not img_path.exists():
                invalid_paths.append(img_path)
        
        if invalid_paths:
            logging.warning(f"Found {len(invalid_paths)} invalid paths")
            self.image_paths = [(p, l) for p, l in self.image_paths if p.exists()]

    def __len__(self) -> int:
        """
        Get dataset length
        Returns:
            - length: number of samples in dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get item at index
        Args:
            - idx: index of item to retrieve
        Returns:
            - sample: dictionary containing image, label, path, and optionally original_image
        """
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
        
        if self.return_original:
            original_image = image.copy() if hasattr(image, 'copy') else image
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        sample = {
            "image": image,
            "label": label,
            "path": str(img_path)
        }
        
        if self.return_original:
            sample["original_image"] = original_image
            
        return sample
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get distribution of classes in dataset
        Returns:
            - distribution: dictionary mapping class indices to counts
        """
        class_counts = {}
        for _, label in self.image_paths:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    
    def get_image_paths_by_class(self, class_idx: int) -> List[Path]:
        """
        Get all image paths for a specific class
        Args:
            - class_idx: class index to filter by
        Returns:
            - paths: list of image paths belonging to the specified class
        """
        return [path for path, label in self.image_paths if label == class_idx]


class CustomDataset(BaseDataset):
    """
    Custom dataset for image classification
    Inherits all functionality from BaseDataset
    """
    pass


class MemoryEfficientDataset(BaseDataset):
    """
    Memory efficient dataset that loads images on-the-fly
    Forces cache_images=False to minimize memory usage
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize MemoryEfficientDataset
        Args:
            - *args: arguments for BaseDataset
            - **kwargs: keyword arguments for BaseDataset (cache_images forced to False)
        """
        # Force cache_images to False for memory efficiency
        kwargs['cache_images'] = False
        super().__init__(*args, **kwargs)