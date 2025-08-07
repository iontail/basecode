from pathlib import Path
from typing import Dict, List, Tuple


def get_class_mapping(data_dir: Path) -> Dict[str, int]:
    """Create class name to integer mapping from directory structure."""
    class_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return {class_name: idx for idx, class_name in enumerate(class_dirs)}


def collect_image_paths(data_dir: Path) -> List[Tuple[Path, int]]:
    """Collect all image paths with their labels."""
    class_mapping = get_class_mapping(data_dir)
    image_paths = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    for class_name, label in class_mapping.items():
        class_dir = data_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in valid_extensions:
                    image_paths.append((img_path, label))
    
    return image_paths