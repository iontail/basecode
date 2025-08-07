from pathlib import Path
from typing import Dict, List, Tuple


def get_class_mapping(data_dir: Path) -> Dict[str, int]:
    """Create class name to integer mapping from directory structure."""
    class_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return {class_name: idx for idx, class_name in enumerate(class_dirs)}

"""
- this functions collects all image paths and their corresponding labels
- collect the specific extensions you want to include
"""
def collect_image_paths(data_dir: str, extensions: List[str] = None) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    data_dir = Path(data_dir)
    class_mapping = get_class_mapping(data_dir)
    image_paths = []
    extensions = [ext.lower() for ext in extensions]

    for class_name, label in class_mapping.items():
        class_dir = data_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in extensions:
                    image_paths.append((img_path, label))

    return image_paths, class_mapping