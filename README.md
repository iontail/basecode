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
Implement the required methods as needed
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
        return {'loss': total_loss,
                'loss1': loss1,
                'loss2': loss2}
    
    def validate_epoch(self, val_loader):
        # Validation loop
        return {'val_loss': total_avg_loss,
                'loss1': loss1,
                'loss2': loss2}
```

## ✨ Features

- **Training**: Mixed precision, multi-GPU support, gradient clipping, early stopping
- **Scheduling**: Cosine annealing, step, plateau schedulers with warmup support
- **Logging**: WandB and TensorBoard integration with automatic metrics tracking
- **Data Pipeline**: Advanced augmentations, balanced sampling, memory-efficient loading
- **Checkpointing**: Best model tracking, automatic cleanup, resume capability
- **Architecture**: Abstract base classes with modular, extensible design

## 📋 Requirements

**Models** must inherit from `BaseModel` and implement:

- `dim` property: Output dimension - **REQUIRED**
- `forward()`: Forward pass - **REQUIRED**  
- `init_weights()`: Weight initialization - **REQUIRED** (if specific initialization is needed)

**Trainers** must inherit from `BaseTrainer` and implement:
- `get_criterion()`: Loss function(s) - **REQUIRED**
- `train_epoch()`: Training loop - **REQUIRED**
- `validate_epoch()`: Validation loop - **REQUIRED**
- `forward_pass()`: Forward pass with loss computation - **REQUIRED**

## 🔧 Usage Examples

```bash
# Basic training
python train.py --data_path ./dataset --epochs 100

# Advanced training with features
python train.py --mixed_precision --use_wandb --save_best \
    --experiment_name "my_exp" --lr 1e-3 --batch_size 64

# Resume training
python train.py --resume ./checkpoints/best_model.pt
```

## 🐛 Bug Reports & Issues
Found a bug or have questions? Please email leechanhye@g.skku.edu

## 📄 License
This project is freely available for all research purposes