import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.metrics import MetricsCalculator, top_k_accuracy, calculate_class_weights
from utils.checkpoint import CheckpointManager, create_checkpoint_from_trainer
from utils.logger import MetricTracker
from trainer.base_trainer import BaseTrainer
from models.base_models import BaseModel
import torch.nn as nn


class MockModel(BaseModel):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(10, num_classes)
    
    @property
    def dim(self) -> int:
        return self.num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class MockTrainer:
    def __init__(self):
        self.model = MockModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = None
        self.scaler = None
        
        class MockArgs:
            pass
        self.args = MockArgs()


class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 3
        self.class_names = ['class1', 'class2', 'class3']
        self.calculator = MetricsCalculator(
            num_classes=self.num_classes,
            class_names=self.class_names
        )
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        # Create mock predictions and targets
        predictions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 1, 1, 2, 0, 2])
        probabilities = torch.softmax(torch.randn(8, 3), dim=1)
        
        # Update calculator
        self.calculator.update(predictions, targets, probabilities)
        
        # Compute metrics
        metrics = self.calculator.compute()
        
        # Check that all expected metrics are present
        expected_keys = ['accuracy', 'precision', 'recall', 'f1']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)
    
    def test_confusion_matrix(self):
        """Test confusion matrix generation"""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 1, 1, 2])
        
        self.calculator.update(predictions, targets)
        cm = self.calculator.get_confusion_matrix()
        
        # Check shape
        self.assertEqual(cm.shape, (self.num_classes, self.num_classes))
        
        # Check that sum equals number of samples
        self.assertEqual(cm.sum(), len(predictions))
    
    def test_reset_functionality(self):
        """Test metrics reset"""
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        
        self.calculator.update(predictions, targets)
        self.assertEqual(len(self.calculator.predictions), 3)
        
        self.calculator.reset()
        self.assertEqual(len(self.calculator.predictions), 0)
        self.assertEqual(len(self.calculator.targets), 0)
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation"""
        # Perfect predictions
        predictions = torch.tensor([[0.9, 0.1, 0.0],
                                  [0.1, 0.8, 0.1],
                                  [0.0, 0.2, 0.8]])
        targets = torch.tensor([0, 1, 2])
        
        # Top-1 accuracy should be 1.0
        acc_1 = top_k_accuracy(predictions, targets, k=1)
        self.assertEqual(acc_1, 1.0)
        
        # Top-2 accuracy should also be 1.0
        acc_2 = top_k_accuracy(predictions, targets, k=2)
        self.assertEqual(acc_2, 1.0)
    
    def test_class_weights_calculation(self):
        """Test class weights calculation"""
        # Imbalanced dataset
        targets = [0] * 10 + [1] * 5 + [2] * 20
        
        weights = calculate_class_weights(targets, method='inverse')
        
        # Check that minority class has higher weight
        self.assertGreater(weights[1], weights[0])  # Class 1 (5 samples) > Class 0 (10 samples)
        self.assertGreater(weights[1], weights[2])  # Class 1 (5 samples) > Class 2 (20 samples)


class TestCheckpointManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            max_checkpoints=3,
            save_best=True,
            monitor_metric='val_loss',
            mode='min'
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        # Create mock state dict
        state_dict = {'param1': torch.randn(10), 'param2': torch.randn(5)}
        metrics = {'val_loss': 0.5, 'accuracy': 0.8}
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            state_dict=state_dict,
            epoch=1,
            metrics=metrics
        )
        
        # Check file exists
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Check history updated
        self.assertEqual(len(self.checkpoint_manager.checkpoint_history), 1)
    
    def test_load_checkpoint(self):
        """Test checkpoint loading"""
        # Save a checkpoint first
        state_dict = {'param1': torch.randn(10)}
        metrics = {'val_loss': 0.3}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            state_dict=state_dict,
            epoch=5,
            metrics=metrics
        )
        
        # Load checkpoint
        loaded_checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Verify content
        self.assertEqual(loaded_checkpoint['epoch'], 5)
        self.assertEqual(loaded_checkpoint['metrics'], metrics)
        self.assertIn('state_dict', loaded_checkpoint)
    
    def test_best_checkpoint_tracking(self):
        """Test best checkpoint tracking"""
        # Save checkpoints with different metrics
        checkpoints = [
            ({'param1': torch.randn(5)}, 1, {'val_loss': 0.8}),
            ({'param1': torch.randn(5)}, 2, {'val_loss': 0.5}),  # Best
            ({'param1': torch.randn(5)}, 3, {'val_loss': 0.7})
        ]
        
        for state_dict, epoch, metrics in checkpoints:
            self.checkpoint_manager.save_checkpoint(
                state_dict=state_dict,
                epoch=epoch,
                metrics=metrics
            )
        
        # Check best metric was tracked
        self.assertEqual(self.checkpoint_manager.best_metric, 0.5)
        
        # Check best checkpoint file exists
        best_path = Path(self.temp_dir) / "best_checkpoint.pt"
        self.assertTrue(best_path.exists())
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup"""
        # Save more checkpoints than max_checkpoints
        for i in range(5):
            state_dict = {'param1': torch.randn(5)}
            metrics = {'val_loss': 0.5 + i * 0.1}
            
            self.checkpoint_manager.save_checkpoint(
                state_dict=state_dict,
                epoch=i,
                metrics=metrics
            )
        
        # Should only keep max_checkpoints regular checkpoints
        regular_checkpoints = [
            cp for cp in self.checkpoint_manager.checkpoint_history
            if not cp.get('is_best', False)
        ]
        self.assertLessEqual(len(regular_checkpoints), self.checkpoint_manager.max_checkpoints)
    
    def test_create_checkpoint_from_trainer(self):
        """Test creating checkpoint from trainer"""
        trainer = MockTrainer()
        
        checkpoint = create_checkpoint_from_trainer(
            trainer,
            epoch=10,
            metrics={'loss': 0.5}
        )
        
        # Verify checkpoint structure
        expected_keys = ['epoch', 'model_state_dict', 'metrics', 'args']
        for key in expected_keys:
            self.assertIn(key, checkpoint)
        
        self.assertEqual(checkpoint['epoch'], 10)
        self.assertEqual(checkpoint['metrics']['loss'], 0.5)


class TestMetricTracker(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = MetricTracker('loss', 'accuracy', 'f1')
    
    def test_metric_update_and_log(self):
        """Test metric updating and logging"""
        # Update metrics
        self.tracker.update(loss=0.5, accuracy=0.8)
        self.tracker.log()
        
        # Check values were logged
        self.assertEqual(len(self.tracker.metrics['loss']), 1)
        self.assertEqual(len(self.tracker.metrics['accuracy']), 1)
        self.assertEqual(self.tracker.metrics['loss'][0], 0.5)
        self.assertEqual(self.tracker.metrics['accuracy'][0], 0.8)
    
    def test_average_calculation(self):
        """Test average calculation"""
        # Add multiple values
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for val in values:
            self.tracker.update(loss=val)
            self.tracker.log()
        
        # Test full average
        avg_all = self.tracker.get_average('loss')
        expected_avg = sum(values) / len(values)
        self.assertAlmostEqual(avg_all, expected_avg, places=6)
        
        # Test last N average
        avg_last_3 = self.tracker.get_average('loss', last_n=3)
        expected_avg_3 = sum(values[-3:]) / 3
        self.assertAlmostEqual(avg_last_3, expected_avg_3, places=6)
    
    def test_best_value_tracking(self):
        """Test best value tracking"""
        values = [0.5, 0.3, 0.7, 0.2, 0.4]
        for val in values:
            self.tracker.update(loss=val)
            self.tracker.log()
        
        # Test minimum (best for loss)
        best_min = self.tracker.get_best('loss', mode='min')
        self.assertEqual(best_min, min(values))
        
        # Test maximum
        best_max = self.tracker.get_best('loss', mode='max')
        self.assertEqual(best_max, max(values))
    
    def test_history_retrieval(self):
        """Test history retrieval"""
        values = [0.1, 0.2, 0.3]
        for val in values:
            self.tracker.update(loss=val)
            self.tracker.log()
        
        history = self.tracker.get_history('loss')
        self.assertEqual(history, values)
    
    def test_reset_functionality(self):
        """Test reset functionality"""
        self.tracker.update(loss=0.5)
        self.tracker.log()
        
        # Verify data exists
        self.assertEqual(len(self.tracker.metrics['loss']), 1)
        
        # Reset and verify empty
        self.tracker.reset()
        self.assertEqual(len(self.tracker.metrics['loss']), 0)
        self.assertEqual(len(self.tracker.current_values), 0)


if __name__ == '__main__':
    unittest.main()