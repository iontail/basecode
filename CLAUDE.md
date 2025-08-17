# CLAUDE.md

Claude Code guidance for PyTorch deep learning framework.

## Project Overview

Modular PyTorch research template with comprehensive training infrastructure, data pipeline, and experiment management.

## Code Style Guidelines

- **Concise, clear code** - Avoid unnecessary complexity
- **Core functionality focus** - Make components easy to understand
- **Minimal comments** - Only for genuinely difficult code
- **Readable over clever** - Prioritize clarity
- **Simple first** - Start straightforward, add complexity later

## Core Architecture

- **BaseTrainer**: `src/trainer/base_trainer.py` - Training infrastructure (mixed precision, logging, checkpointing, early stopping)
- **BaseModel**: `src/models/base_models.py` - Models must implement `forward()`, `dim` property, `init_weights()` method
- **Data Pipeline**: `src/data/` - Dataset loading, augmentation, preprocessing
- **Configuration**: `arguments.py` - CLI arguments with validation

## Key Files to Customize

1. `src/models/model.py` - Model architecture (inherit from BaseModel)
2. `src/trainer/main_trainer.py` - Training logic (inherit from BaseTrainer)
3. `src/data/dataset.py` - Dataset classes and preprocessing
4. `arguments.py` - CLI arguments and hyperparameters

## Development Commands

```bash
# Setup
pip install uv && uv sync && source .venv/bin/activate

# Training
python train.py --data_path ./dataset --epochs 100 --batch_size 32

# Testing
python test.py --weights ./checkpoints/best_model.pth

# View arguments
python arguments.py
```

## Implementation Pattern

### 1. Define Model (BaseModel)
```python
class YourModel(BaseModel):
    def init_weights(self):  # Required
        # Initialize all weights
        pass
    
    @property
    def dim(self):  # Required
        return self.output_dim
    
    def forward(self, x):  # Required
        return self.layers(x)
```

### 2. Create Trainer (BaseTrainer)
```python
class MainTrainer(BaseTrainer):
    def get_criterion(self):  # Required
        return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):  # Required
        # Training loop
        return {'loss': avg_loss}
    
    def validate_epoch(self, val_loader):  # Required
        # Validation loop
        return {'val_loss': avg_val_loss}
    
    def forward_pass(self, batch):  # Required
        return loss, predictions
    
    def inference(self, batch):  # Required
        return predictions
```

## Key Features

- **Training**: Mixed precision, multi-GPU, gradient clipping, early stopping
- **Scheduling**: Cosine, step, plateau, exponential with warmup support
- **Logging**: WandB, TensorBoard with automatic metrics tracking
- **Data**: Advanced augmentations, balanced sampling, memory-efficient loading
- **Checkpointing**: Best model tracking, automatic cleanup, resume capability
- **Loss Functions**: Single/multiple/named loss support for multi-task learning
- **Reproducibility**: Deterministic training with seed management

## Loss Function Flexibility

```python
# Single loss
return nn.CrossEntropyLoss()

# Multiple losses
return (nn.CrossEntropyLoss(), nn.MSELoss())
return [nn.CrossEntropyLoss(), nn.MSELoss()]

# Named losses (multi-task)
return {'classification': nn.CrossEntropyLoss(), 'regression': nn.MSELoss()}
```

## Experiment Management

### Logging
```bash
# WandB tracking
python train.py --use_wandb --wandb_project "my_project" --experiment_name "exp1"

# TensorBoard
python train.py --use_tensorboard

# Reproducible experiments
python train.py --seed 42 --deterministic
```

### Checkpointing
- Auto-saves to `./checkpoints/`
- Best model tracking via validation loss
- Resume with `--resume checkpoint.pt`
- Includes optimizer/scheduler/scaler states

### Advanced Features

```bash
# Mixed precision training
python train.py --mixed_precision

# Multi-GPU training
python train.py --use_multigpu

# Advanced scheduling with warmup
python train.py --scheduler cosine --warmup_epochs 10

# Early stopping
python train.py --early_stopping --patience 15

# Model compilation (PyTorch 2.0+)
python train.py --compile
```

## Common Usage Patterns

### Systematic Experiments
```bash
# Learning rate search
for lr in 1e-4 1e-3 1e-2; do
    python train.py --lr $lr --experiment_name "lr_${lr}"
done

# Architecture comparison
for model in unet resnet18 resnet50; do
    python train.py --model $model --experiment_name "${model}_baseline"
done
```

### Production Training
```bash
# Full featured training
python train.py \
    --mixed_precision \
    --use_wandb \
    --save_best \
    --early_stopping \
    --patience 20 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --experiment_name "production_run"
```

## Troubleshooting

### Common Issues
- **OOM**: `--batch_size 16 --mixed_precision`
- **Slow training**: `--compile --num_workers 8`
- **Poor convergence**: `--lr 1e-4 --warmup_epochs 10`
- **Overfitting**: `--dropout 0.2 --weight_decay 1e-4`

### Debugging
- Use `--debug` for small data subset
- Check data loading with `--fast_dev_run`
- Monitor gradients with `--grad_clip`
- Profile with `--profile`

## Implementation Reminders

- **Models**: Must implement `init_weights()`, `dim` property, `forward()`
- **Trainers**: Must implement `get_criterion()`, `train_epoch()`, `validate_epoch()`, `forward_pass()`, `inference()`
- **Data Loaders**: Pass as parameters to training methods for flexibility
- **Loss Functions**: Support single/multiple/named patterns for multi-task learning
- **Weight Initialization**: Called automatically during trainer setup
- **Checkpointing**: Includes complete training state for seamless resuming

## 🚨 Critical Restrictions (MUST FOLLOW)

### 🔴 Database Commands - ABSOLUTELY FORBIDDEN
```bash
# Database destructive commands - NEVER use without explicit user permission
reset commands...

# SQL destructive commands - ABSOLUTELY FORBIDDEN
DROP, TRUNCATE, DELETE, ALTER
```

### 🛡️ Database Operation Rules
1. **ALWAYS request explicit user permission before data deletion/reset**
2. **NEVER delete data without backup**
3. **Additional rules to be added**

### 🔴 Git Dangerous Commands - ABSOLUTELY FORBIDDEN
```bash
git push --force
git reset --hard
git commit --no-verify
```

### 🔴 NPM Dangerous Commands
```bash
npm audit fix --force
```

### Library Version Lock (DO NOT CHANGE)