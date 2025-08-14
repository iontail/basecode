import unittest
import torch
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import BaseDataset, CustomDataset, MemoryEfficientDataset
from data.loader import get_transforms, create_dataloader
from data.utils import collect_image_paths, create_balanced_split, analyze_dataset_balance
from data.collate import collate_fn, mixup_collate_fn


class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_data"
        
        # Create test directory structure
        self.create_test_dataset()
        
        # Mock args object
        class MockArgs:
            def __init__(self):
                self.resize = 32
                self.crop_size = 32
                self.random_crop = True
                self.random_flip = True
                self.normalize = True
                self.mean = [0.5, 0.5, 0.5]
                self.std = [0.5, 0.5, 0.5]
                self.augmentation = True
                self.auto_augment = 'none'
                self.color_jitter = False
                self.random_rotation = False
                self.gaussian_blur = False
                self.num_workers = 0
                self.pin_memory = False
                self.memory_efficient = False
                self.balanced_sampling = False
                self.mixup = False
        
        self.args = MockArgs()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self):
        """Create a test dataset with multiple classes"""
        # Create class directories
        classes = ['class1', 'class2', 'class3']
        
        for cls in classes:
            cls_dir = self.data_dir / cls
            cls_dir.mkdir(parents=True)
            
            # Create different number of images per class to test balance
            num_images = {'class1': 10, 'class2': 5, 'class3': 15}[cls]
            
            for i in range(num_images):
                # Create random RGB image
                img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = cls_dir / f"image_{i:03d}.jpg"
                img.save(img_path)
    
    def test_collect_image_paths(self):
        """Test image path collection"""
        image_paths, class_mapping = collect_image_paths(self.data_dir)
        
        # Check that all images were found
        self.assertEqual(len(image_paths), 30)  # 10 + 5 + 15
        self.assertEqual(len(class_mapping), 3)
        
        # Check class mapping
        expected_classes = {'class1', 'class2', 'class3'}
        self.assertEqual(set(class_mapping.keys()), expected_classes)
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        image_paths, class_mapping = collect_image_paths(self.data_dir)
        
        dataset = CustomDataset(image_paths)
        
        # Test dataset length
        self.assertEqual(len(dataset), 30)
        
        # Test item access
        item = dataset[0]
        self.assertIn('image', item)
        self.assertIn('label', item)
        self.assertIn('path', item)
        
        # Test image is PIL Image
        self.assertIsInstance(item['image'], Image.Image)
    
    def test_transforms(self):
        """Test transform pipeline"""
        transform = get_transforms(self.args, is_train=True)
        
        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        transformed = transform(img)
        
        # Check output is tensor
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, (3, 32, 32))
        
        # Test validation transforms
        val_transform = get_transforms(self.args, is_train=False)
        val_transformed = val_transform(img)
        self.assertIsInstance(val_transformed, torch.Tensor)
    
    def test_balanced_split(self):
        """Test balanced data splitting"""
        image_paths, _ = collect_image_paths(self.data_dir)
        split_ratios = [0.7, 0.2, 0.1]
        
        train_paths, val_paths, test_paths = create_balanced_split(image_paths, split_ratios)
        
        # Check split sizes are approximately correct
        total_size = len(image_paths)
        self.assertAlmostEqual(len(train_paths) / total_size, 0.7, delta=0.1)
        self.assertAlmostEqual(len(val_paths) / total_size, 0.2, delta=0.1)
        self.assertAlmostEqual(len(test_paths) / total_size, 0.1, delta=0.1)
        
        # Check no overlap between splits
        train_set = set(p[0] for p in train_paths)
        val_set = set(p[0] for p in val_paths)
        test_set = set(p[0] for p in test_paths)
        
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)
    
    def test_dataset_balance_analysis(self):
        """Test dataset balance analysis"""
        image_paths, _ = collect_image_paths(self.data_dir)
        stats = analyze_dataset_balance(image_paths)
        
        self.assertEqual(stats['total_samples'], 30)
        self.assertEqual(stats['num_classes'], 3)
        self.assertEqual(stats['min_samples_per_class'], 5)
        self.assertEqual(stats['max_samples_per_class'], 15)
        self.assertEqual(stats['imbalance_ratio'], 3.0)  # 15/5
    
    def test_collate_functions(self):
        """Test collate functions"""
        # Create mock batch
        batch = []
        for i in range(4):
            batch.append({
                'image': torch.randn(3, 32, 32),
                'label': i % 2,
                'path': f'path_{i}'
            })
        
        # Test standard collate
        collated = collate_fn(batch)
        self.assertEqual(collated['images'].shape, (4, 3, 32, 32))
        self.assertEqual(collated['labels'].shape, (4,))
        self.assertEqual(len(collated['paths']), 4)
        
        # Test mixup collate
        mixup_batch = mixup_collate_fn(batch, alpha=0.2)
        self.assertEqual(mixup_batch['images'].shape, (4, 3, 32, 32))
        self.assertIn('labels_b', mixup_batch)
        self.assertIn('lam', mixup_batch)
    
    def test_dataloader_creation(self):
        """Test dataloader creation"""
        # Create train subdirectory
        train_dir = self.data_dir / "train"
        train_dir.mkdir()
        
        # Move some images to train directory
        for cls in ['class1', 'class2']:
            cls_train_dir = train_dir / cls
            cls_train_dir.mkdir()
            
            src_dir = self.data_dir / cls
            for i, img_file in enumerate(src_dir.glob("*.jpg")):
                if i < 3:  # Move first 3 images
                    shutil.copy2(img_file, cls_train_dir / img_file.name)
        
        # Test dataloader creation
        dataloader = create_dataloader(
            train_dir,
            self.args,
            batch_size=2,
            shuffle=True,
            is_train=True
        )
        
        # Test batch
        batch = next(iter(dataloader))
        self.assertEqual(batch['images'].shape[0], 2)  # batch size
        self.assertEqual(batch['images'].shape[1], 3)  # channels
        self.assertEqual(len(batch['paths']), 2)
    
    def test_memory_efficient_dataset(self):
        """Test memory efficient dataset"""
        image_paths, _ = collect_image_paths(self.data_dir)
        
        # Test that memory efficient dataset doesn't cache
        dataset = MemoryEfficientDataset(image_paths)
        self.assertFalse(dataset.cache_images)
        self.assertIsNone(dataset.image_cache)
        
        # Should still work like normal dataset
        item = dataset[0]
        self.assertIn('image', item)
        self.assertIn('label', item)


if __name__ == '__main__':
    unittest.main()