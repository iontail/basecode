import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculator for various classification metrics"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """Update metrics with batch predictions"""
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
        
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probs is not None:
            if probs.dim() > 1 and probs.size(1) > 1:
                probs = torch.softmax(probs, dim=1)
            self.probabilities.extend(probs.cpu().numpy())
    
    def compute(self, average: str = 'weighted') -> Dict[str, float]:
        """Compute all metrics"""
        if not self.predictions:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average=average, zero_division=0),
            'recall': recall_score(targets, preds, average=average, zero_division=0),
            'f1': f1_score(targets, preds, average=average, zero_division=0)
        }
        
        # Add per-class metrics
        if average == 'weighted':
            per_class_precision = precision_score(targets, preds, average=None, zero_division=0)
            per_class_recall = recall_score(targets, preds, average=None, zero_division=0)
            per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
            
            for i in range(min(len(per_class_precision), self.num_classes)):
                metrics[f'precision_class_{i}'] = per_class_precision[i]
                metrics[f'recall_class_{i}'] = per_class_recall[i]
                metrics[f'f1_class_{i}'] = per_class_f1[i]
        
        # ROC AUC for binary or multi-class with probabilities
        if self.probabilities and len(self.probabilities) > 0:
            probs = np.array(self.probabilities)
            try:
                if self.num_classes == 2:
                    if probs.ndim == 2:
                        probs = probs[:, 1]  # Use positive class probability
                    metrics['auc'] = roc_auc_score(targets, probs)
                elif self.num_classes > 2 and probs.ndim == 2:
                    metrics['auc'] = roc_auc_score(targets, probs, multi_class='ovr', average=average)
            except ValueError:
                # Skip AUC if not possible to compute
                pass
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.predictions:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(
            self.targets, 
            self.predictions, 
            labels=range(self.num_classes)
        )
    
    def plot_confusion_matrix(self, normalize: bool = False, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot confusion matrix"""
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names[:cm.shape[1]], 
            yticklabels=self.class_names[:cm.shape[0]],
            ax=ax
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        return fig
    
    def classification_report(self) -> str:
        """Get detailed classification report"""
        if not self.predictions:
            return "No predictions available"
        
        return classification_report(
            self.targets, 
            self.predictions, 
            target_names=self.class_names,
            zero_division=0
        )


def top_k_accuracy(preds: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Calculate top-k accuracy"""
    if preds.dim() == 1:
        return float(preds.eq(targets).float().mean())
    
    _, top_k_preds = preds.topk(k, dim=1)
    correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
    return float(correct.any(dim=1).float().mean())


def calculate_class_weights(targets: List[int], method: str = 'inverse') -> torch.Tensor:
    """Calculate class weights for imbalanced datasets"""
    from collections import Counter
    
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    total_samples = len(targets)
    
    if method == 'inverse':
        weights = torch.zeros(num_classes)
        for cls, count in class_counts.items():
            weights[cls] = total_samples / (num_classes * count)
    elif method == 'sqrt_inverse':
        weights = torch.zeros(num_classes)
        for cls, count in class_counts.items():
            weights[cls] = np.sqrt(total_samples / count)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights