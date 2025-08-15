# 🔬 PyTorch Deep Learning Framework

Modular PyTorch template for deep learning research with comprehensive training infrastructure, data pipeline, and experiment management.

## 🚀 Quick Start

### Setup

```bash
git clone https://github.com/iontail/basecode.git
cd basecode

# UV (recommended)
pip install uv && uv sync && source .venv/bin/activate

# Or Conda
conda create -n basecode python=3.12
conda activate basecode
pip install -r requirements.txt
```

### Key Files

| File | Purpose |
|------|----------|
| `src/models/model.py` | Model architecture |
| `src/data/dataset.py` | Dataset & preprocessing |
| `src/trainer/main_trainer.py` | Training logic |
| `arguments.py` | CLI arguments |

### Usage

```bash
# Training
python train.py --data_path ./dataset --epochs 100 --batch_size 32

# Testing
python test.py --weights ./checkpoints/best_model.pth
```

## 📁 Structure

```
src/
├── data/           # Dataset, loading, preprocessing
├── models/         # Model architectures
├── trainer/        # Training infrastructure  
└── utils/          # Utilities, metrics

train.py           # Training entry
test.py            # Evaluation entry
arguments.py       # CLI arguments
```

## 🛠️ Implementation

### Model
```python
from .base_models import BaseModel

class YourModel(BaseModel):
    def init_weights(self):
        # Required: initialize weights
        pass
    
    @property
    def dim(self):
        # Required: output dimension
        return self.output_dim
    
    def forward(self, x):
        return self.layers(x)
```

### Trainer
```python
from .base_trainer import BaseTrainer

class MainTrainer(BaseTrainer):
    def get_criterion(self):
        return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        # Training loop
        return {'loss': avg_loss}
    
    def validate_epoch(self, val_loader):
        # Validation loop
        return {'val_loss': avg_val_loss}
```

## ✨ Features

- **Training**: Mixed precision, multi-GPU, gradient clipping
- **Scheduling**: Cosine annealing, step, plateau with warmup  
- **Logging**: WandB, TensorBoard integration
- **Data**: Advanced augmentations, balanced sampling
- **Checkpointing**: Best model tracking, resume training
- **Architecture**: Abstract base classes, modular design

## 📋 Requirements

**Models** must inherit from `BaseModel` and implement:

- `dim` property: Output dimension
- `forward()`: Forward pass
- `init_weights()`: Weight initialization, if need specific initialization

**Trainers** must inherit from `BaseTrainer` and implement:
- `get_criterion()`: Loss function(s)
- `train_epoch()`: Training loop
- `validate_epoch()`: Validation loop

## 🔧 Examples

```bash
# Basic training
python train.py --data_path ./dataset --epochs 100

# Advanced training
python train.py --mixed_precision --use_wandb --save_best \
    --experiment_name "my_exp" --lr 1e-3 --batch_size 64

# Resume training
python train.py --resume ./checkpoints/best_model.pt
```

## 📞 Contact

**Issues**: leechanhye@g.skku.edu

**License**: Open for research use