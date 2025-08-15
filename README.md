# 🔬 PyTorch Research Template

A clean, modular template for deep learning research. Get started quickly with computer vision, audio, or multimodal projects.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/iontail/basecode.git
cd basecode

# Option 1: UV (Recommended)
pip install uv
uv sync
source .venv/bin/activate

# Option 2: Conda
conda create -n basecode python=3.12
conda activate basecode
pip install -r requirements.txt
```

### 2. Customize for Your Research

Edit these key files:

| File | What to Change |
|------|----------------|
| `src/models/model.py` | Your model architecture |
| `src/data/dataset.py` | Dataset loading & preprocessing |
| `src/trainer/main_trainer.py` | Training loop logic |
| `arguments.py` | Add your hyperparameters |

### 3. Run Training

```bash
# UV environment
uv run python train.py --train_data_dir ./your_dataset --epochs 100
uv run python test.py --weights ./checkpoints/best_model.pth

# Traditional environment
python train.py --train_data_dir ./your_dataset --epochs 100
python test.py --weights ./checkpoints/best_model.pth
```

## 📁 Project Structure

```
src/
├── data/           # Dataset & data loading
├── models/         # Model definitions
├── trainer/        # Training logic
└── utils/          # Logging, metrics, etc.

train.py           # Training entry point
test.py            # Evaluation entry point
arguments.py       # CLI arguments
```

## 🛠️ Implementation Guide

### 1. Define Your Model
```python
# src/models/model.py
from .base_models import BaseModel

class YourModel(BaseModel):
    def forward(self, x):
        # Your forward pass
        return output
```

### 2. Create Dataset Class
```python
# src/data/dataset.py
from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(self, data_dir):
        # Load your data
        pass
    
    def __getitem__(self, idx):
        # Return sample
        return data, label
```

### 3. Customize Training
```python
# src/trainer/main_trainer.py
from .base_trainer import BaseTrainer
import torch.nn as nn

class MainTrainer(BaseTrainer):
    def get_criterion(self):
        # Define loss function(s): single, tuple, list, or dict
        return nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
    
    def train_epoch(self, train_loader):
        # Training logic with data loader as parameter
        return {'loss': avg_loss}
    
    def validate_epoch(self, val_loader):
        # Validation logic with data loader as parameter
        return {'val_loss': avg_val_loss}
```

## ✨ Features

- Mixed precision training (AMP)
- Flexible data loader parameters
- Multiple loss function support
- Automatic checkpointing & resume
- WandB/TensorBoard logging
- Multi-GPU support
- Learning rate scheduling
- Early stopping

## 🐛 Issues & Support

Found a bug or have questions? Please contact me at **leechanhye@g.skku.edu**

## 📄 License

Open for research use.