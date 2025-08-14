from pathlib import Path
from typing import Dict, List, Tuple, Union
import random
from collections import defaultdict, Counter
import numpy as np


def get_class_mapping(data_dir: Path) -> Dict[str, int]:
    """Get mapping from class names to indices"""
    class_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return {class_name: idx for idx, class_name in enumerate(class_dirs)}


def collect_image_paths(
    data_dir: Union[str, Path], 
    extensions: List[str] = None,
    recursive: bool = True
) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    """Collect all image paths with their labels"""
    data_dir = Path(data_dir)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    extensions = [ext.lower() for ext in extensions]
    image_paths = []
    
    # Check if data_dir has subdirectories (class-based organization)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        # Class-based organization
        class_mapping = get_class_mapping(data_dir)
        
        for class_name, label in class_mapping.items():
            class_dir = data_dir / class_name
            if class_dir.exists():
                search_pattern = "**/*" if recursive else "*"
                for img_path in class_dir.glob(search_pattern):
                    if img_path.is_file() and img_path.suffix.lower() in extensions:
                        image_paths.append((img_path, label))
    else:
        # Flat organization - assign all images to class 0
        class_mapping = {"default": 0}
        search_pattern = "**/*" if recursive else "*"
        for img_path in data_dir.glob(search_pattern):
            if img_path.is_file() and img_path.suffix.lower() in extensions:
                image_paths.append((img_path, 0))

    print(f"Found {len(image_paths)} images in {len(class_mapping)} classes")
    return image_paths, class_mapping


def create_balanced_split(
    image_paths: List[Tuple[Path, int]], 
    split_ratios: List[float],
    seed: int = 42
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Create balanced train/val/test split"""
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must have exactly 3 values [train, val, test]")
    
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError("split_ratios must sum to 1.0")
    
    # Group by class
    class_paths = defaultdict(list)
    for path, label in image_paths:
        class_paths[label].append((path, label))
    
    # Split each class
    random.seed(seed)
    train_paths, val_paths, test_paths = [], [], []
    
    for label, paths in class_paths.items():
        random.shuffle(paths)
        n_total = len(paths)
        
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        n_test = n_total - n_train - n_val
        
        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train:n_train + n_val])
        test_paths.extend(paths[n_train + n_val:])
    
    # Shuffle the final splits
    random.shuffle(train_paths)
    random.shuffle(val_paths) 
    random.shuffle(test_paths)
    
    return train_paths, val_paths, test_paths


def analyze_dataset_balance(image_paths: List[Tuple[Path, int]]) -> Dict[str, Union[int, float]]:
    """Analyze class distribution in dataset"""
    labels = [label for _, label in image_paths]
    class_counts = Counter(labels)
    
    total = len(labels)
    num_classes = len(class_counts)
    
    # Calculate statistics
    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    
    # Imbalance ratio
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'total_samples': total,
        'num_classes': num_classes,
        'min_samples_per_class': min_count,
        'max_samples_per_class': max_count,
        'mean_samples_per_class': mean_count,
        'std_samples_per_class': std_count,
        'imbalance_ratio': imbalance_ratio,
        'class_distribution': dict(class_counts)
    }


def create_class_weights(image_paths: List[Tuple[Path, int]], method: str = 'inverse') -> Dict[int, float]:
    """Create class weights for handling imbalanced datasets"""
    labels = [label for _, label in image_paths]
    class_counts = Counter(labels)
    
    if method == 'inverse':
        # Inverse frequency weighting
        total = len(labels)
        weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    elif method == 'sqrt_inverse':
        # Square root inverse frequency
        total = len(labels)
        weights = {cls: np.sqrt(total / count) for cls, count in class_counts.items()}
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights


def filter_by_class_size(
    image_paths: List[Tuple[Path, int]], 
    min_samples: int = 10,
    max_samples: int = None
) -> List[Tuple[Path, int]]:
    """Filter classes by minimum and maximum sample count"""
    class_counts = Counter(label for _, label in image_paths)
    
    valid_classes = set()
    for cls, count in class_counts.items():
        if count >= min_samples:
            if max_samples is None or count <= max_samples:
                valid_classes.add(cls)
    
    filtered_paths = [(path, label) for path, label in image_paths if label in valid_classes]
    
    print(f"Filtered from {len(set(class_counts.keys()))} to {len(valid_classes)} classes")
    print(f"Samples: {len(image_paths)} -> {len(filtered_paths)}")
    
    return filtered_paths