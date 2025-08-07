from typing import List, Dict
import torch

def default_collate_fn(batch: List[Dict]) -> Dict:
    """
    배치 단위로 dict 구성: 'image', 'label', 'path' 등을 하나로 묶음
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    paths = [item["path"] for item in batch]

    return {
        "images": images,   # B×C×H×W
        "labels": labels,   # B
        "paths": paths      # list-of-str
    }
