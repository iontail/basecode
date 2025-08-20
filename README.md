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

# Training with loss weights
python train.py --ce_weight 0.7 --focal_weight 0.3 --focal_alpha 1.0

# Testing
python test.py --weights ./checkpoints/best_model.pth
```

## 📁 Structure

```
src/
├── data/           # Dataset, loading, preprocessing
├── models/         # Model architectures
├── trainer/        # Training infrastructure  
└── utils/          # Utilities, metrics, loss functions

train.py           # Training entry
test.py            # Evaluation entry
arguments.py       # CLI arguments
```

## 🔬 Research Workflow

### Step-by-Step Guide for Starting Your Research

#### 1. Define Your Loss Function (`src/utils/loss.py`)
**MODIFY FIRST** - This determines your training objective
```python
def get_loss_function(args):
    # For classification
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # For regression
    return nn.MSELoss()
    
    # For custom loss with parameters from args
    return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    # For combined losses with weights from args
    return CombinedLoss({
        'ce': nn.CrossEntropyLoss(),
        'focal': FocalLoss(gamma=2.0)
    }, weights={
        'ce': args.ce_weight,
        'focal': args.focal_weight
    })
```

**Configure loss weights via command line:**
```bash
# Single loss with parameters
python train.py --focal_alpha 1.0 --focal_gamma 2.0

# Combined loss weights
python train.py --ce_weight 0.7 --focal_weight 0.3

# Multi-task loss weights
python train.py --classification_weight 1.0 --regression_weight 0.5
```

#### 2. Implement Your Model (`src/models/model.py`)
**Core component** - Replace UNet with your architecture
```python
class YourModel(BaseModel):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        # Your architecture here
        self.layers = nn.Sequential(...)
        self.num_classes = num_classes
        
    def init_weights(self):  # REQUIRED
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    @property
    def dim(self):  # REQUIRED
        return self.num_classes
    
    def forward(self, x):  # REQUIRED
        return self.layers(x)
```

#### 3. Configure Data Pipeline (`src/data/`)
**Customize for your dataset**

**`dataset.py`** - Define your data loading:
```python
# Modify BaseDataset or create CustomDataset
class YourDataset(BaseDataset):
    def __init__(self, data_path, transform=None, **kwargs):
        # Load your data paths and labels
        image_paths = self.load_data_paths(data_path)
        super().__init__(image_paths, transform, **kwargs)
```

**`loader.py`** - Adjust data loading parameters

#### 4. Customize Training Logic (`src/trainer/main_trainer.py`) 
**Optional** - Only if you need custom training loops
```python
class MainTrainer(BaseTrainer):
    # Loss is automatically loaded from src/utils/loss.py
    
    def train_epoch(self, train_loader):  # Override if needed
        # Custom training logic
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self, val_loader):  # Override if needed
        # Custom validation logic
        return {'val_loss': avg_loss, 'val_accuracy': accuracy}
```

#### 5. Update Arguments (`arguments.py`)
**Configure hyperparameters** for your specific task:
```python
# Modify model-specific arguments
parser.add_argument('--model', default='your_model')

# ... rest of code
```

#### 6. Update Training Entry (`train.py`)
**Connect your model** to the training pipeline:
```python
def main():
    args = parse_arguments()
    
    if args.model == 'your_model':
        model = YourModel(
            num_classes=args.num_classes,
            input_channels=args.input_channels
        )
    # ... rest of training code
```



### Essential Modifications Summary

| Component | File | What to Modify |
|-----------|------|----------------|
| **Loss Function** | `src/utils/loss.py` | `get_loss_function()` - Define your training objective |
| **Model Architecture** | `src/models/model.py` | Replace UNet, implement required methods |
| **Dataset** | `src/data/dataset.py` | Customize data loading for your data format |
| **Hyperparameters** | `arguments.py` | Add model/task-specific arguments |
| **Training Entry** | `train.py` | Connect your model to training pipeline |
| **Training Logic** | `src/trainer/main_trainer.py` | *(Optional)* Custom training loops |

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