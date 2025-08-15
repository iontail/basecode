import torch
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime


class CheckpointManager:
    """Manager for handling model checkpoints"""
    
    def __init__(
        self, 
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor_metric: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Initialize CheckpointManager
        Args:
            - checkpoint_dir: directory to save checkpoints
            - max_checkpoints: maximum number of regular checkpoints to keep
            - save_best: whether to save best checkpoint separately
            - monitor_metric: metric to monitor for best checkpoint
            - mode: 'min' or 'max' for best metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        
        self.checkpoint_history: List[Dict[str, Any]] = []
    
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint with model state and training information
        Args:
            - state_dict: model and training state dictionary
            - epoch: current training epoch
            - metrics: optional dictionary of metrics for this epoch
            - is_best: whether this is marked as best checkpoint
            - filename: optional custom filename for checkpoint
        Returns:
            - checkpoint_path: path where checkpoint was saved
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'state_dict': state_dict,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update history
        self.checkpoint_history.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics or {},
            'is_best': is_best
        })
        
        # Check if this is the best checkpoint
        if self.save_best and metrics and self.monitor_metric in metrics:
            metric_value = metrics[self.monitor_metric]
            
            is_better = (
                (self.mode == 'min' and metric_value < self.best_metric) or
                (self.mode == 'max' and metric_value > self.best_metric)
            )
            
            if is_better:
                self.best_metric = metric_value
                self.best_checkpoint_path = str(checkpoint_path)
                
                # Save as best checkpoint
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                shutil.copy2(checkpoint_path, best_path)
                
                logging.info(f"New best checkpoint saved with {self.monitor_metric}={metric_value:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self, 
        checkpoint_path: Optional[str] = None, 
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from file
        Args:
            - checkpoint_path: specific checkpoint path to load (optional)
            - load_best: whether to load the best checkpoint
        Returns:
            - checkpoint: loaded checkpoint dictionary
        """
        
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        elif checkpoint_path is None:
            # Load latest checkpoint
            checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint
        Returns:
            - path: path to latest checkpoint or None if no checkpoints exist
        """
        if not self.checkpoint_history:
            # Search directory for checkpoints
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoint_files:
                return None
            
            # Sort by modification time
            latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            return str(latest_file)
        
        # Get from history
        latest = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return latest['path']
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get the path to the best checkpoint
        Returns:
            - path: path to best checkpoint or None if no best checkpoint exists
        """
        return self.best_checkpoint_path
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all saved checkpoints with their metadata
        Returns:
            - checkpoints: list of checkpoint information dictionaries
        """
        return self.checkpoint_history.copy()
    
    def _cleanup_checkpoints(self):
        """
        Remove old checkpoints if exceeding maximum limit
        Keeps the most recent checkpoints and preserves best checkpoint
        """
        if self.max_checkpoints <= 0:
            return
        
        # Get non-best checkpoints
        regular_checkpoints = [
            cp for cp in self.checkpoint_history 
            if not cp.get('is_best', False)
        ]
        
        if len(regular_checkpoints) > self.max_checkpoints:
            # Sort by epoch and keep only the most recent
            regular_checkpoints.sort(key=lambda x: x['epoch'])
            checkpoints_to_remove = regular_checkpoints[:-self.max_checkpoints]
            
            for checkpoint_info in checkpoints_to_remove:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logging.info(f"Removed old checkpoint: {checkpoint_path}")
                
                # Remove from history
                self.checkpoint_history.remove(checkpoint_info)
    
    def delete_checkpoint(self, checkpoint_path: str):
        """
        Delete a specific checkpoint file and remove from history
        Args:
            - checkpoint_path: path to checkpoint file to delete
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logging.info(f"Deleted checkpoint: {checkpoint_path}")
        
        # Remove from history
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history 
            if cp['path'] != str(checkpoint_path)
        ]
    
    def export_checkpoint(self, checkpoint_path: str, export_path: str):
        """
        Export checkpoint to a different location (copy)
        Args:
            - checkpoint_path: source checkpoint path
            - export_path: destination path for exported checkpoint
        """
        src_path = Path(checkpoint_path)
        dst_path = Path(export_path)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source checkpoint not found: {src_path}")
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        
        logging.info(f"Exported checkpoint from {src_path} to {dst_path}")


def create_checkpoint_from_trainer(
    trainer,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    include_optimizer: bool = True,
    include_scheduler: bool = True
) -> Dict[str, Any]:
    """
    Create checkpoint dictionary from trainer object
    Args:
        - trainer: trainer object with model, optimizer, scheduler
        - epoch: current training epoch
        - metrics: optional metrics dictionary
        - include_optimizer: whether to include optimizer state
        - include_scheduler: whether to include scheduler state
    Returns:
        - checkpoint: complete checkpoint dictionary
    """
    
    # Handle DataParallel/DDP models
    model_state = trainer.model.module.state_dict() if hasattr(trainer.model, 'module') else trainer.model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'metrics': metrics or {},
        'args': trainer.args
    }
    
    if include_optimizer and trainer.optimizer:
        checkpoint['optimizer_state_dict'] = trainer.optimizer.state_dict()
    
    if include_scheduler and trainer.scheduler:
        checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
    
    if hasattr(trainer, 'scaler') and trainer.scaler:
        checkpoint['scaler_state_dict'] = trainer.scaler.state_dict()
    
    return checkpoint


def load_checkpoint_to_trainer(trainer, checkpoint: Dict[str, Any], load_optimizer: bool = True):
    """
    Load checkpoint data into trainer object
    Args:
        - trainer: trainer object to load checkpoint into
        - checkpoint: checkpoint dictionary to load
        - load_optimizer: whether to load optimizer state
    Returns:
        - tuple: (epoch, metrics) from loaded checkpoint
    """
    
    # Load model state
    if hasattr(trainer.model, 'module'):
        trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if load_optimizer and trainer.optimizer and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state
    if hasattr(trainer, 'scaler') and trainer.scaler and 'scaler_state_dict' in checkpoint:
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})