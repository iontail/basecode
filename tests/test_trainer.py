import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trainer.base_trainer import BaseTrainer
from models.base_models import BaseModel


class MockModel(BaseModel):
    """Mock model for testing"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(16, num_classes)
    
    @property
    def dim(self) -> int:
        return self.num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)
    
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class MockTrainer(BaseTrainer):
    """Mock trainer for testing"""
    
    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()
    
    def train_epoch(self) -> dict:
        return {'loss': 1.0, 'accuracy': 0.8}
    
    def validate_epoch(self) -> dict:
        return {'val_loss': 0.9, 'val_accuracy': 0.85}
    
    def forward_pass(self, batch):
        images = batch['images']
        labels = batch['labels']
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss, outputs
    
    def inference(self, batch):
        images = batch['images']
        with torch.no_grad():
            return self.model.inference(images)


class TestTrainer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock args
        class MockArgs:
            def __init__(self):
                self.device = 'cpu'
                self.seed = 42
                self.deterministic = True
                self.save_dir = self.temp_dir
                self.log_dir = self.temp_dir
                self.experiment_name = 'test_experiment'
                
                # Training args
                self.epochs = 5
                self.lr = 0.001
                self.weight_decay = 0.0001
                self.optimizer = 'AdamW'
                self.scheduler = 'cosine'
                self.min_lr = 1e-6
                
                # Mixed precision
                self.mixed_precision = False
                self.grad_clip = 1.0
                
                # Logging
                self.use_tensorboard = False
                self.use_wandb = False
                
                # Multi-GPU
                self.use_multigpu = False
                self.compile = False
                
                # Checkpointing
                self.save_best = True
                self.save_freq = 2
                self.keep_checkpoint_max = 3
                self.early_stopping = False
                self.patience = 10
                
                # Evaluation
                self.eval_freq = 1
        
        self.args = MockArgs()
        self.model = MockModel(num_classes=10)
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = MockTrainer(self.model, self.args)
        
        # Check device setup
        self.assertEqual(trainer.device, torch.device('cpu'))
        
        # Check model is moved to device
        self.assertEqual(next(trainer.model.parameters()).device, trainer.device)
        
        # Check attributes
        self.assertEqual(trainer.current_epoch, 0)
        self.assertIsNone(trainer.best_metric)
        self.assertEqual(trainer.early_stopping_counter, 0)
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        trainer = MockTrainer(self.model, self.args)
        optimizer = trainer.get_optimizer()
        
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], self.args.lr)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], self.args.weight_decay)
    
    def test_scheduler_creation(self):
        """Test scheduler creation"""
        trainer = MockTrainer(self.model, self.args)
        optimizer = trainer.get_optimizer()
        scheduler = trainer.get_scheduler(optimizer)
        
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, self.args.epochs)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        trainer = MockTrainer(self.model, self.args)
        trainer.optimizer = trainer.get_optimizer()
        trainer.scheduler = trainer.get_scheduler(trainer.optimizer)
        trainer.criterion = trainer.get_criterion()
        
        # Set some training state
        trainer.current_epoch = 5
        trainer.best_metric = 0.95
        
        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), is_best=True)
        
        # Check file exists
        self.assertTrue(checkpoint_path.exists())
        
        # Create new trainer and load checkpoint
        new_trainer = MockTrainer(MockModel(num_classes=10), self.args)
        new_trainer.optimizer = new_trainer.get_optimizer()
        new_trainer.scheduler = new_trainer.get_scheduler(new_trainer.optimizer)
        
        checkpoint = new_trainer.load_checkpoint(str(checkpoint_path))
        
        # Check state was loaded
        self.assertEqual(new_trainer.current_epoch, 5)
        self.assertEqual(new_trainer.best_metric, 0.95)
    
    def test_model_summary(self):
        """Test model summary generation"""
        trainer = MockTrainer(self.model, self.args)
        
        # This should not raise an exception
        trainer.model_summary()
        
        # Test model size calculation
        size_bytes = trainer.model_size_b(trainer.model)
        self.assertGreater(size_bytes, 0)
    
    def test_metrics_logging(self):
        """Test metrics logging"""
        trainer = MockTrainer(self.model, self.args)
        
        metrics = {
            'train': {'loss': 1.0, 'accuracy': 0.8},
            'val': {'val_loss': 0.9, 'val_accuracy': 0.85}
        }
        
        # This should not raise an exception
        trainer.log_metrics(metrics, epoch=1)
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        trainer = MockTrainer(self.model, self.args)
        trainer.optimizer = trainer.get_optimizer()
        
        # Create some gradients
        dummy_input = torch.randn(2, 3, 32, 32)
        dummy_target = torch.randint(0, 10, (2,))
        
        output = trainer.model(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_target)
        loss.backward()
        
        # Test gradient clipping
        trainer.clip_gradients()
        
        # Check that gradients are within bounds
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm()
                self.assertLessEqual(grad_norm, self.args.grad_clip + 1e-6)  # Small tolerance
    
    def test_mixed_precision_step(self):
        """Test mixed precision optimization step"""
        # Test without mixed precision (scaler = None)
        trainer = MockTrainer(self.model, self.args)
        trainer.optimizer = trainer.get_optimizer()
        trainer.scaler = None
        
        dummy_loss = torch.tensor(1.0, requires_grad=True)
        
        # This should not raise an exception
        trainer.mixed_precision_step(dummy_loss)
        
        # Test with mixed precision
        trainer.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        if trainer.scaler:
            trainer.mixed_precision_step(dummy_loss)
    
    def test_learning_rate_getter(self):
        """Test learning rate getter"""
        trainer = MockTrainer(self.model, self.args)
        
        # Without optimizer
        self.assertEqual(trainer.get_lr(), 0.0)
        
        # With optimizer
        trainer.optimizer = trainer.get_optimizer()
        self.assertEqual(trainer.get_lr(), self.args.lr)


class TestBaseModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = MockModel(num_classes=10)
    
    def test_model_properties(self):
        """Test model properties"""
        self.assertEqual(self.model.dim, 10)
    
    def test_freeze_unfreeze(self):
        """Test parameter freezing"""
        # Initially all parameters should be trainable
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(trainable_before, total_params)
        
        # Freeze all parameters
        self.model.freeze()
        trainable_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(trainable_after, 0)
        
        # Unfreeze all parameters
        self.model.unfreeze()
        trainable_final = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(trainable_final, total_params)
    
    def test_selective_freeze(self):
        """Test selective parameter freezing"""
        # Freeze only classifier layers
        self.model.freeze(['classifier'])
        
        # Check that classifier parameters are frozen
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)
    
    def test_parameter_counting(self):
        """Test parameter counting"""
        total, trainable = self.model.count_parameters()
        
        self.assertGreater(total, 0)
        self.assertEqual(total, trainable)  # Initially all parameters are trainable
        
        # After freezing
        self.model.freeze()
        total_after, trainable_after = self.model.count_parameters()
        
        self.assertEqual(total_after, total)  # Total should remain same
        self.assertEqual(trainable_after, 0)  # No trainable parameters
    
    def test_device_operations(self):
        """Test device operations"""
        device = self.model.get_device()
        self.assertEqual(device, torch.device('cpu'))
        
        # Test device movement
        self.model.to_device('cpu')
        self.assertEqual(self.model.get_device(), torch.device('cpu'))
    
    def test_trainable_parameters(self):
        """Test getting trainable parameters"""
        trainable_params = self.model.get_trainable_parameters()
        total_trainable = sum(p.numel() for p in trainable_params)
        
        expected_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_trainable, expected_trainable)


if __name__ == '__main__':
    unittest.main()