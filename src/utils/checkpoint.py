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
        """Save a checkpoint"""
        
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
        """Load a checkpoint"""
        
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
        """Get the path to the latest checkpoint"""
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
        """Get the path to the best checkpoint"""
        return self.best_checkpoint_path
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        return self.checkpoint_history.copy()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints if exceeding max limit"""
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
        """Delete a specific checkpoint"""
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
        """Export checkpoint to a different location"""
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
    """Create checkpoint dictionary from trainer"""
    
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
    """Load checkpoint data to trainer"""
    
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