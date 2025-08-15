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
- **Abstract Model Interface**: Models inherit from `BaseModel` (in `src/models/base_models.py`) and must implement `forward()`, `dim` property, and `init_weights()` method
- **Configuration System**: Uses `arguments.py` for CLI arguments + `config/config.py` for dataset-specific configurations (means, stds, class counts)
- **Data Pipeline**: `src/data/` contains modular dataset loading, collation, and preprocessing utilities

### Key Files to Customize for New Projects
1. `src/models/model.py` - Implement your model architecture inheriting from BaseModel (must implement `init_weights()` method)
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
   - Must implement abstract methods: `forward()`, `dim` property, and `init_weights()`
   - The `init_weights()` method is automatically called during model initialization
2. **Create Custom Trainer**: Inherit from `BaseTrainer` and implement:
   - `get_criterion()` - Define loss function(s) (single/tuple/list/dict)
   - `train_epoch(train_loader)` - Training loop logic with data loader parameter
   - `validate_epoch(val_loader)` - Validation logic with data loader parameter
   - `forward_pass()` - Forward pass with loss computation
   - `inference()` - Inference logic
3. **Configure Data**: Update `config/config.py` with your dataset parameters
4. **Run Training**: Use `train.py` entry point with your arguments

## Training Features

- **Flexible Architecture**: Data loaders passed as parameters for maximum flexibility
- **Multiple Loss Support**: Single loss function, tuple/list of losses, or dict of named losses
- **Mixed Precision**: Automatic FP16 optimization (AMP) support
- **Multi-GPU Training**: DataParallel/DDP support
- **Comprehensive Logging**: WandB, TensorBoard integration with real-time metrics
- **Smart Checkpointing**: Best model tracking, automatic saving, and resume capability
- **Early Stopping**: Configurable patience with validation-based termination
- **Advanced Scheduling**: Cosine, step, plateau, exponential LR schedulers
- **Data Augmentation**: AutoAugment, CutMix, Mixup pipeline support
- **Model Compilation**: PyTorch 2.0+ optimization
- **Reproducibility**: Comprehensive seed setting and deterministic controls
- **Automatic Weight Initialization**: All models must implement `init_weights()` method for proper initialization

## Checkpointing System

- Automatic checkpoint saving to `./checkpoints/`
- Best model tracking based on validation loss
- Resume training with `--resume path/to/checkpoint.pt`
- Includes optimizer, scheduler, and scaler states

## Experiment Management

### Logging and Tracking
- **WandB Integration**: Set `--use_wandb` to enable comprehensive experiment tracking
  - Automatic hyperparameter logging
  - Real-time metrics visualization
  - Model artifact storage
  - Custom project naming with `--wandb_project`
- **TensorBoard Support**: Use `--use_tensorboard` for local visualization
- **Progress Tracking**: Built-in tqdm progress bars for epoch monitoring

### Experiment Organization
```bash
# Structured experiment naming
python train.py --experiment_name "resnet50_cifar10_baseline" --use_wandb
python train.py --experiment_name "resnet50_cifar10_augmented" --use_wandb

# Reproducible experiments
python train.py --seed 42 --deterministic --experiment_name "reproducible_run"
```

### Model Evaluation and Analysis
- **Comprehensive Model Summary**: Automatic parameter counting and memory usage
- **Inference Pipeline**: Built-in inference methods for model evaluation
- **Checkpoint Analysis**: Load and compare different model checkpoints
- **Performance Profiling**: Memory and compute efficiency monitoring

### Advanced Training Techniques
- **Mixed Precision Training**: Automatic FP16 optimization with `--mixed_precision`
- **Gradient Clipping**: Prevent exploding gradients with `--grad_clip`
- **Learning Rate Scheduling**: Multiple scheduler options (cosine, step, plateau, exponential)
- **Early Stopping**: Intelligent training termination with `--early_stopping --patience N`
- **Model Compilation**: PyTorch 2.0+ optimization with `--compile`

### Hyperparameter Optimization
```bash
# Learning rate experiments
python train.py --lr 0.001 --experiment_name "lr_0001"
python train.py --lr 0.01 --experiment_name "lr_001"
python train.py --lr 0.1 --experiment_name "lr_01"

# Batch size scaling
python train.py --batch_size 32 --lr 0.001 --experiment_name "bs32_lr0001"
python train.py --batch_size 64 --lr 0.002 --experiment_name "bs64_lr0002"
```

## Research Workflow Best Practices

### 1. Baseline Establishment
```bash
# Start with simple baseline
python train.py --model simple_cnn --epochs 50 --experiment_name "baseline"
```

### 2. Systematic Experimentation
- One variable at a time approach
- Document all changes in experiment names
- Use consistent evaluation metrics
- Track computational resources

### 3. Model Comparison
```bash
# Compare architectures
python train.py --model resnet18 --experiment_name "resnet18_comparison"
python train.py --model resnet50 --experiment_name "resnet50_comparison"
python train.py --model efficientnet_b0 --experiment_name "efficientnet_comparison"
```

### 4. Hyperparameter Search Strategy
- **Grid Search**: Systematic parameter exploration
- **Random Search**: Efficient parameter sampling
- **Learning Rate Range Test**: Find optimal learning rates
- **Batch Size Scaling**: Match batch size with learning rate

### 5. Validation and Testing
- **Cross-validation**: Multiple data splits for robust evaluation
- **Hold-out Test Set**: Final model evaluation on unseen data
- **Statistical Significance**: Multiple runs with different seeds
- **Error Analysis**: Detailed failure case examination

## Performance Optimization

### Memory Optimization
- **Mixed Precision**: Reduce memory usage by 50%
- **Gradient Checkpointing**: Trade compute for memory
- **Batch Size Scheduling**: Dynamic batch size adjustment
- **Model Pruning**: Remove redundant parameters

### Compute Optimization
- **Multi-GPU Training**: Scale with `--use_multigpu`
- **Model Compilation**: PyTorch 2.0+ speedups
- **Data Loading**: Optimized DataLoader settings
- **Profiling**: Identify bottlenecks

### Data Pipeline Optimization
- **Prefetching**: Parallel data loading
- **Caching**: Store preprocessed data
- **Augmentation**: Balanced augmentation strategies
- **Batch Collation**: Efficient batch creation

## Debugging and Troubleshooting

### Common Issues
- **Out of Memory**: Reduce batch size, enable mixed precision
- **Slow Training**: Check data loading, enable compilation
- **Poor Convergence**: Adjust learning rate, check loss function
- **Overfitting**: Add regularization, reduce model complexity

### Debugging Tools
- **Gradient Monitoring**: Track gradient norms
- **Learning Curve Analysis**: Plot training/validation metrics
- **Model Visualization**: Understand model behavior
- **Data Inspection**: Verify data quality and preprocessing

## Advanced Research Features

### Model Architecture Search
- **Neural Architecture Search (NAS)**: Automated architecture discovery
- **Progressive Growing**: Gradually increase model complexity
- **Transfer Learning**: Leverage pre-trained models
- **Multi-task Learning**: Joint training on multiple objectives

### Specialized Training Regimes
- **Curriculum Learning**: Progressive difficulty scheduling
- **Self-supervised Learning**: Learn from unlabeled data
- **Few-shot Learning**: Adapt to new tasks with minimal data
- **Meta-learning**: Learn to learn quickly

## Trainer Pattern and Data Flow

### Core Training Loop Architecture
The training system follows a clean separation of concerns:

```python
# BaseTrainer handles the orchestration
def train(self):
    for epoch in range(epochs):
        train_metrics = self.train_epoch(self.train_loader)  # Pass loader as parameter
        val_metrics = self.validate_epoch(self.val_loader)   # Pass loader as parameter
        self.log_metrics({'train': train_metrics, 'val': val_metrics})

# MainTrainer implements the specifics
class MainTrainer(BaseTrainer):
    def train_epoch(self, train_loader):
        # Your training logic with the provided loader
        for batch in train_loader:
            loss, predictions = self.forward_pass(batch)
            # Training step...
        return metrics
```

### Loss Function Flexibility
The `get_criterion()` method supports multiple patterns:
- **Single Loss**: `return nn.CrossEntropyLoss()`
- **Multiple Losses (Tuple)**: `return (nn.CrossEntropyLoss(), nn.MSELoss())`
- **Multiple Losses (List)**: `return [nn.CrossEntropyLoss(), nn.MSELoss()]`
- **Named Losses (Dict)**: `return {'classification': nn.CrossEntropyLoss(), 'regression': nn.MSELoss()}`

This flexibility allows for complex training scenarios like multi-task learning or composite loss functions.

### Data Loader Parameter Pattern
By passing data loaders as parameters rather than using instance variables, the system gains:
- **Testing Flexibility**: Easy to test with mock data loaders
- **Runtime Flexibility**: Different loaders for different scenarios (e.g., different augmentation strategies)
- **Multi-dataset Support**: Switch between datasets within the same training session
- **Clean Separation**: Training logic is independent of data source specifics
