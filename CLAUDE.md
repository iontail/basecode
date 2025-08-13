# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular PyTorch deep learning research template designed for rapid prototyping and structured experimentation. The codebase supports various computer vision tasks including classification, object detection, and multimodal learning.

## Code Style Guidelines

**IMPORTANT**: When working on this project, follow these principles:

- **Write concise, clear code** - Avoid unnecessarily long or complex implementations
- **Focus on core functionality** - Make it easy to understand how key components work
- **Minimal comments** - Only add comments for code that is genuinely difficult to understand or interpret
- **Readable over clever** - Prioritize code clarity over optimization tricks
- **Simple implementations first** - Start with straightforward solutions before adding complexity

## Key Architecture Components

### Core Structure
- **Base Trainer System**: `src/trainer/base_trainer.py` contains `BaseTrainer` - an abstract class that provides comprehensive training infrastructure including mixed precision, logging, checkpointing, and early stopping
- **Abstract Model Interface**: Models inherit from `BaseModel` (in `src/models/base_models.py`) and must implement `forward()`, `inference()`, and `dim` property
- **Configuration System**: Uses `arguments.py` for CLI arguments + `config/config.py` for dataset-specific configurations (means, stds, class counts)
- **Data Pipeline**: `src/data/` contains modular dataset loading, collation, and preprocessing utilities

### Key Files to Customize for New Projects
1. `src/models/model.py` - Implement your model architecture inheriting from BaseModel
2. `src/trainer/main_trainer.py` - Create task-specific trainer inheriting from BaseTrainer
3. `src/data/dataset.py` - Define custom dataset classes
4. `config/config.py` - Add dataset configurations (num_classes, normalization values, etc.)
5. `arguments.py` - Modify CLI arguments as needed for your task

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -y -n basecode python=3.12
conda activate basecode

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Training & Testing
```bash
# Training
python train.py --train_data_dir ./dataset/YourDataset --batch_size 32 --epochs 100

# Testing  
python test.py --weights ./weights/your_model.pth

# With configuration arguments
python arguments.py  # View all available arguments
```

### Key Arguments Pattern
The system uses a comprehensive argument parser that automatically handles:
- Device selection (auto-detects CUDA/MPS/CPU)
- Experiment naming with timestamps if not provided
- Directory creation for checkpoints and logs
- Validation of arguments (e.g., split ratios must sum to 1.0)

## Implementation Pattern for New Tasks

1. **Define Your Model**: Inherit from `BaseModel` in `src/models/model.py`
2. **Create Custom Trainer**: Inherit from `BaseTrainer` and implement:
   - `get_criterion()` - Define loss function
   - `train_epoch()` - Training loop logic
   - `validate_epoch()` - Validation logic  
   - `forward_pass()` - Forward pass with loss computation
   - `inference()` - Inference logic
3. **Configure Data**: Update `config/config.py` with your dataset parameters
4. **Run Training**: Use `train.py` entry point with your arguments

## Training Features

- Mixed precision training (AMP) support
- Multi-GPU training (DataParallel/DDP) 
- Comprehensive logging (WandB, TensorBoard)
- Checkpoint management with best model saving
- Early stopping with configurable patience
- Learning rate scheduling (cosine, step, plateau, exponential)
- Data augmentation pipeline (including AutoAugment, CutMix, Mixup)
- Model compilation (PyTorch 2.0+) 
- Reproducibility controls with seed setting

## Checkpointing System

- Automatic checkpoint saving to `./checkpoints/`
- Best model tracking based on validation loss
- Resume training with `--resume path/to/checkpoint.pt`
- Includes optimizer, scheduler, and scaler states