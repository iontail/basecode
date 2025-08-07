# src/data/utils.py

from pathlib import Path
from typing import Dict, List, Tuple

def get_label_map(data_dir: Path) -> Dict[str, int]:
    """
    하위 디렉토리 이름을 기준으로 정수 라벨 매핑 생성
    예: {"cat": 0, "dog": 1, ...}
    """
    class_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return {class_name: idx for idx, class_name in enumerate(class_dirs)}


def get_image_label_list(data_dir: Path, label_map: Dict[str, int]) -> List[Tuple[Path, int]]:
    """
    (이미지 경로, 정수 라벨) 튜플 리스트 생성
    - 하위 디렉토리명을 클래스명으로 간주
    - 확장자 필터링
    """
    image_label_list = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_name, label in label_map.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob("*"):
            if path.suffix.lower() in valid_exts:
                image_label_list.append((path, label))

    return image_label_list
