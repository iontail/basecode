from typing import List, Dict
import torch
import numpy as np


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Standard collate function for batching dataset samples
    Args:
        - batch: list of sample dictionaries from dataset
    Returns:
        - batch_dict: dictionary with batched tensors and metadata
    """
    result = {
        "images": torch.stack([item["image"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "paths": [item["path"] for item in batch]
    }
    
    # Add original images if available
    if "original_image" in batch[0]:
        result["original_images"] = [item["original_image"] for item in batch]
    
    return result


def mixup_collate_fn(batch: List[Dict], alpha: float = 0.2) -> Dict:
    """
    Collate function with Mixup data augmentation
    Args:
        - batch: list of sample dictionaries from dataset
        - alpha: mixup interpolation parameter
    Returns:
        - batch_dict: dictionary with mixed images and dual labels
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    paths = [item["path"] for item in batch]
    
    batch_size = images.size(0)
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        result = {
            "images": mixed_images,
            "labels": labels_a,
            "labels_b": labels_b,
            "lam": lam,
            "paths": paths
        }
        
        # Add original images if available
        if "original_image" in batch[0]:
            result["original_images"] = [item["original_image"] for item in batch]
        
        return result
    else:
        return collate_fn(batch)


def cutmix_collate_fn(batch: List[Dict], alpha: float = 1.0) -> Dict:
    """
    Collate function with CutMix data augmentation
    Args:
        - batch: list of sample dictionaries from dataset
        - alpha: cutmix strength parameter
    Returns:
        - batch_dict: dictionary with cut-mixed images and dual labels
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    paths = [item["path"] for item in batch]
    
    batch_size = images.size(0)
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(batch_size)
        
        labels_a = labels
        labels_b = labels[rand_index]
        
        # Generate random bounding box
        W = images.size(3)
        H = images.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to the exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        result = {
            "images": images,
            "labels": labels_a,
            "labels_b": labels_b,
            "lam": lam,
            "paths": paths
        }
        
        # Add original images if available
        if "original_image" in batch[0]:
            result["original_images"] = [item["original_image"] for item in batch]
        
        return result
    else:
        return collate_fn(batch)


def adaptive_collate_fn(batch: List[Dict], augmentation_type: str = "none", **kwargs) -> Dict:
    """
    Adaptive collate function that chooses augmentation based on type
    Args:
        - batch: list of sample dictionaries from dataset
        - augmentation_type: type of augmentation ('none', 'mixup', 'cutmix')
        - **kwargs: additional arguments for augmentation functions
    Returns:
        - batch_dict: dictionary with appropriately augmented batch
    """
    if augmentation_type == "mixup":
        return mixup_collate_fn(batch, kwargs.get("alpha", 0.2))
    elif augmentation_type == "cutmix":
        return cutmix_collate_fn(batch, kwargs.get("alpha", 1.0))
    else:
        return collate_fn(batch)