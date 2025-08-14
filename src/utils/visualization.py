import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
from torchvision.utils import make_grid
import seaborn as sns
from pathlib import Path


def denormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Denormalize a tensor using mean and std"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def tensor_to_image(tensor: torch.Tensor, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
    """Convert tensor to numpy image"""
    if mean is not None and std is not None:
        tensor = denormalize_tensor(tensor, mean, std)
    
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor[0]
    
    if tensor.size(0) == 1:  # Grayscale
        tensor = tensor.squeeze(0)
        return tensor.cpu().numpy()
    elif tensor.size(0) == 3:  # RGB
        tensor = tensor.permute(1, 2, 0)
        return tensor.cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def plot_batch(
    images: torch.Tensor, 
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    max_images: int = 16,
    figsize: Tuple[int, int] = (12, 8),
    nrow: int = 4
) -> plt.Figure:
    """Plot a batch of images with labels and predictions"""
    
    # Limit number of images
    if images.size(0) > max_images:
        images = images[:max_images]
        if labels is not None:
            labels = labels[:max_images]
        if predictions is not None:
            predictions = predictions[:max_images]
    
    # Create grid
    if mean is not None and std is not None:
        images_denorm = torch.stack([
            denormalize_tensor(img, mean, std) for img in images
        ])
    else:
        images_denorm = images
    
    grid = make_grid(images_denorm, nrow=nrow, normalize=True, pad_value=1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display grid
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    ax.imshow(grid_np)
    ax.axis('off')
    
    # Add titles with labels and predictions
    if labels is not None or predictions is not None:
        n_images = images.size(0)
        cols = min(nrow, n_images)
        rows = (n_images + cols - 1) // cols
        
        for i in range(n_images):
            row = i // cols
            col = i % cols
            
            title_parts = []
            if labels is not None:
                label_name = class_names[labels[i]] if class_names else f"L:{labels[i]}"
                title_parts.append(f"True: {label_name}")
            
            if predictions is not None:
                pred_name = class_names[predictions[i]] if class_names else f"P:{predictions[i]}"
                title_parts.append(f"Pred: {pred_name}")
            
            if title_parts:
                # Calculate position for title
                img_height = grid_np.shape[0] // rows
                img_width = grid_np.shape[1] // cols
                x = col * img_width + img_width // 2
                y = row * img_height - 10
                
                title = " | ".join(title_parts)
                ax.text(x, y, title, ha='center', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Plot training curves for losses and metrics"""
    
    n_plots = 1
    if train_metrics:
        n_plots += len(train_metrics)
    
    fig, axes = plt.subplots(1, min(n_plots, 3), figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif n_plots == 2:
        axes = axes.flatten()
    
    # Plot losses
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss')
    
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Val Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot metrics
    if train_metrics and len(axes) > 1:
        metric_names = list(train_metrics.keys())
        
        for i, metric_name in enumerate(metric_names[:min(2, len(axes) - 1)]):
            ax = axes[i + 1]
            ax.plot(epochs, train_metrics[metric_name], 'b-', label=f'Train {metric_name}')
            
            if val_metrics and metric_name in val_metrics:
                ax.plot(epochs, val_metrics[metric_name], 'r-', label=f'Val {metric_name}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'Training {metric_name.capitalize()}')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_learning_rate(lr_history: List[float], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot learning rate schedule"""
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(lr_history) + 1)
    ax.plot(epochs, lr_history, 'b-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_feature_maps(
    feature_maps: torch.Tensor,
    max_channels: int = 16,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot feature maps from a layer"""
    
    if feature_maps.dim() == 4:
        feature_maps = feature_maps[0]  # Take first sample in batch
    
    n_channels = min(feature_maps.size(0), max_channels)
    cols = 4
    rows = (n_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_channels):
        row = i // cols
        col = i % cols
        
        ax = axes[row, col]
        feature_map = feature_maps[i].cpu().numpy()
        
        im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplots
    for i in range(n_channels, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def save_plot(fig: plt.Figure, save_path: Union[str, Path], dpi: int = 300):
    """Save plot to file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_model_summary_plot(
    model_params: Dict[str, int],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Create a visualization of model parameter distribution"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Parameter count by layer type
    layer_types = []
    param_counts = []
    
    for name, count in model_params.items():
        layer_types.append(name)
        param_counts.append(count)
    
    # Bar plot
    ax1.bar(range(len(layer_types)), param_counts)
    ax1.set_xticks(range(len(layer_types)))
    ax1.set_xticklabels(layer_types, rotation=45, ha='right')
    ax1.set_ylabel('Parameter Count')
    ax1.set_title('Parameters by Layer')
    
    # Pie chart
    ax2.pie(param_counts, labels=layer_types, autopct='%1.1f%%')
    ax2.set_title('Parameter Distribution')
    
    plt.tight_layout()
    return fig