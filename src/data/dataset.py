from pathlib import Path
from typing import Callable, List, Tuple, Dict, Optional
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        image_label_list: List[Tuple[Path, int]],
        transform: Optional[Callable] = None
    ):
        self.image_label_list = image_label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, idx) -> Dict:
        img_path, label = self.image_label_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # dict 형태로 반환
        return {
            "image": image,              # Tensor (C×H×W)
            "label": label,              # int
            "path": str(img_path)        # 경로 정보 (debug 용도)
        }
