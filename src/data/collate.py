from typing import List, Dict
import torch


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for batching samples."""
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "paths": [item["path"] for item in batch]
    }