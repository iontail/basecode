import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# User-Defined Loss Functions
# =============================================================================
# Define your custom loss functions here. 
# The trainer will automatically look for a function named 'get_loss_function()'

def get_loss_function(args=None):
    """
    Define your loss function here.
    
    Args:
        args: training arguments containing loss weights and parameters
    
    Returns:
        loss_function: A PyTorch loss function or callable
        
    Examples:
        # Single loss
        return nn.CrossEntropyLoss()
        
        # Loss with parameters from args
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        
        # Custom loss with args
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        
        # Combined losses with weights from args
        return CombinedLoss(
            losses={
                'ce': nn.CrossEntropyLoss(),
                'focal': FocalLoss(gamma=2.0)
            },
            weights={
                'ce': args.ce_weight,
                'focal': args.focal_weight
            }
        )
        
        # Multi-task losses with individual weights
        return {
            'classification': nn.CrossEntropyLoss(),
            'regression': nn.MSELoss(),
            'segmentation': DiceLoss()
        }
    """
    # Default loss - modify this for your research
    if args is None:
        return nn.CrossEntropyLoss()
    
    # Use label smoothing from args if available
    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# =============================================================================
# Custom Loss Function Examples
# =============================================================================
# You can define custom loss classes below and use them in get_loss_function()

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with weights
    """
    def __init__(self, losses: dict, weights: Optional[dict] = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses.keys()}
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            total_loss += weight * loss_fn(inputs, targets)
        
        return total_loss


# =============================================================================
# Usage Examples (commented out)
# =============================================================================
"""
Example 1: Using Focal Loss
def get_loss_function():
    return FocalLoss(alpha=1.0, gamma=2.0)

Example 2: Combined Loss with weights from args
def get_loss_function(args):
    return CombinedLoss(
        losses={
            'ce': nn.CrossEntropyLoss(),
            'focal': FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        },
        weights={
            'ce': args.ce_weight,
            'focal': args.focal_weight
        }
    )

Example 3: Multi-task Loss
def get_loss_function():
    return {
        'classification': nn.CrossEntropyLoss(),
        'regression': nn.MSELoss(),
        'segmentation': DiceLoss()
    }

Example 4: Custom Loss Function
class MyCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets):
        # Your custom loss logic
        return loss

def get_loss_function():
    return MyCustomLoss()
"""