import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """Unified logger for training experiments"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        log_level: str = "INFO"
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging(log_level)
        
        # Setup wandb
        self.wandb_enabled = use_wandb and WANDB_AVAILABLE
        if self.wandb_enabled:
            self._setup_wandb(wandb_project, wandb_entity)
        
        # Setup tensorboard
        self.tensorboard_enabled = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.tensorboard_enabled:
            self._setup_tensorboard()
        
        # Metrics storage
        self.metrics_history: Dict[str, list] = {}
        
        logging.info(f"Logger initialized for experiment: {experiment_name}")
        logging.info(f"Wandb enabled: {self.wandb_enabled}")
        logging.info(f"Tensorboard enabled: {self.tensorboard_enabled}")
    
    def _setup_file_logging(self, log_level: str):
        """Setup file logging"""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler  
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def _setup_wandb(self, project: Optional[str], entity: Optional[str]):
        """Setup Weights & Biases logging"""
        try:
            wandb.init(
                project=project or "deep-learning-research",
                entity=entity,
                name=self.experiment_name,
                dir=str(self.log_dir)
            )
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            self.wandb_enabled = False
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging"""
        try:
            tb_log_dir = self.log_dir / "tensorboard" / self.experiment_name
            self.tb_writer = SummaryWriter(str(tb_log_dir))
        except Exception as e:
            logging.warning(f"Failed to initialize TensorBoard: {e}")
            self.tensorboard_enabled = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to all enabled loggers"""
        
        # Add to history
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append(value)
        
        # Log to wandb
        if self.wandb_enabled:
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(log_dict, step=step)
        
        # Log to tensorboard
        if self.tensorboard_enabled:
            for key, value in metrics.items():
                full_key = f"{prefix}/{key}" if prefix else key
                self.tb_writer.add_scalar(full_key, value, step)
        
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        prefix_str = f"{prefix} " if prefix else ""
        logging.info(f"Step {step} | {prefix_str}{metric_str}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        
        # Save to file
        hparams_file = self.log_dir / f"{self.experiment_name}_hparams.json"
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
        
        # Log to wandb
        if self.wandb_enabled:
            wandb.config.update(hparams)
        
        # Log to tensorboard
        if self.tensorboard_enabled:
            self.tb_writer.add_hparams(hparams, {})
        
        logging.info("Hyperparameters logged")
    
    def log_model_summary(self, model_summary: str):
        """Log model architecture summary"""
        
        # Save to file
        summary_file = self.log_dir / f"{self.experiment_name}_model_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(model_summary)
        
        logging.info("Model summary saved")
    
    def log_image(self, tag: str, image, step: int, dataformats: str = "CHW"):
        """Log image to tensorboard and wandb"""
        
        if self.tensorboard_enabled:
            self.tb_writer.add_image(tag, image, step, dataformats=dataformats)
        
        if self.wandb_enabled:
            if hasattr(image, 'numpy'):
                image = image.numpy()
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to tensorboard"""
        
        if self.tensorboard_enabled:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text to tensorboard and wandb"""
        
        if self.tensorboard_enabled:
            self.tb_writer.add_text(tag, text, step)
        
        if self.wandb_enabled:
            wandb.log({tag: text}, step=step)
    
    def save_metrics(self):
        """Save metrics history to file"""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logging.info(f"Metrics saved to {metrics_file}")
    
    def close(self):
        """Close all loggers"""
        
        if self.tensorboard_enabled:
            self.tb_writer.close()
        
        if self.wandb_enabled:
            wandb.finish()
        
        self.save_metrics()
        logging.info("Logger closed")


class MetricTracker:
    """Simple metric tracker for training"""
    
    def __init__(self, *metric_names: str):
        self.metrics = {name: [] for name in metric_names}
        self.current_values = {}
    
    def update(self, **kwargs):
        """Update metric values"""
        for name, value in kwargs.items():
            if name in self.metrics:
                self.current_values[name] = value
    
    def log(self, reset: bool = True):
        """Log current values and optionally reset"""
        for name, value in self.current_values.items():
            if name in self.metrics:
                self.metrics[name].append(value)
        
        if reset:
            self.current_values.clear()
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """Get average of last N values"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = self.metrics[name]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def get_best(self, name: str, mode: str = 'min') -> float:
        """Get best value"""
        if name not in self.metrics or not self.metrics[name]:
            return float('inf') if mode == 'min' else float('-inf')
        
        values = self.metrics[name]
        return min(values) if mode == 'min' else max(values)
    
    def get_history(self, name: str) -> list:
        """Get full history for a metric"""
        return self.metrics.get(name, []).copy()
    
    def reset(self):
        """Reset all metrics"""
        for name in self.metrics:
            self.metrics[name].clear()
        self.current_values.clear()


def setup_logging(
    log_dir: str,
    log_level: str = "INFO",
    experiment_name: Optional[str] = None
):
    """Setup basic logging configuration"""
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log_file = log_dir / f"{experiment_name}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return experiment_name