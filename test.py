#!/usr/bin/env python3
"""
Test script for evaluating trained models on test data.
Loads trained model checkpoints and evaluates performance on test dataset.
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import time
from pathlib import Path
from datetime import datetime

from arguments import parse_arguments
from src.models.model import UNet
from src.trainer.main_trainer import MainTrainer
from src.data.loader import create_dataloader
from src.utils.metrics import MetricsCalculator


def parse_test_arguments():
    """Parse command line arguments for testing"""
    parser = argparse.ArgumentParser(description='Model Testing Configuration')
    
    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test dataset')
    
    # Optional arguments
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'resnet18', 'resnet50', 'vit'],
                        help='Model architecture to use')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for testing')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save individual predictions to file')
    parser.add_argument('--save_confusion_matrix', action='store_true', default=True,
                        help='Save confusion matrix plot')
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                        help='List of class names for reporting')
    
    # Data processing arguments
    parser.add_argument('--resize', type=int, default=224,
                        help='Resize input images to this size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size for images')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize the dataset')
    parser.add_argument('--mean', type=float, nargs=3, default=[0.4914, 0.4822, 0.4465],
                        help='Mean values for normalization')
    parser.add_argument('--std', type=float, nargs=3, default=[0.2023, 0.1994, 0.2010],
                        help='Standard deviation values for normalization')
    parser.add_argument('--extensions', type=list, default=['.jpg', '.png', '.jpeg'],
                        help='List of valid image file extensions')
    
    # Testing options
    parser.add_argument('--tta', action='store_true',
                        help='Use test time augmentation')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output during testing')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup and return the appropriate device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def load_model_and_checkpoint(args, device):
    """Load model architecture and trained weights"""
    print(f"Loading model: {args.model}")
    
    # Create model based on architecture
    if args.model == 'unet':
        model = UNet(
            channels=[32, 64, 128, 256], 
            num_res_blocks=2,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.weights}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"Training metrics: {checkpoint['metrics']}")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def run_inference(model, test_loader, device, args):
    """Run inference on test dataset and collect predictions"""
    print("Running inference on test data...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_paths = []
    
    total_batches = len(test_loader)
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            paths = batch.get('path', [f'sample_{i}' for i in range(len(images))])
            
            # Forward pass
            if args.tta:
                # Test Time Augmentation - simple horizontal flip
                outputs = model(images)
                outputs_flip = model(torch.flip(images, dims=[3]))
                outputs = (outputs + outputs_flip) / 2
            else:
                outputs = model(images)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            # Collect results
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_probabilities.append(probabilities.cpu())
            all_paths.extend(paths)
            
            if args.verbose and (batch_idx + 1) % 10 == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Progress: {progress:.1f}% ({batch_idx + 1}/{total_batches})")
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    
    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time / len(all_predictions) * 1000:.2f} ms")
    
    return all_predictions, all_targets, all_probabilities, all_paths


def calculate_metrics(predictions, targets, probabilities, class_names=None):
    """Calculate comprehensive evaluation metrics"""
    print("Calculating metrics...")
    
    num_classes = len(torch.unique(targets))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(num_classes, class_names)
    metrics_calc.update(predictions, targets, probabilities)
    
    # Calculate all metrics
    metrics = metrics_calc.compute()
    
    return metrics, metrics_calc


def save_results(args, metrics, metrics_calc, predictions, targets, paths, probabilities):
    """Save test results to files"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to JSON
    metrics_file = output_dir / f"test_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        # Convert numpy values to float for JSON serialization
        json_metrics = {k: float(v) for k, v in metrics.items()}
        json.dump({
            'test_results': json_metrics,
            'model': args.model,
            'checkpoint': args.weights,
            'test_data': args.data_path,
            'num_samples': len(predictions),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    # Save detailed classification report
    report_file = output_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write("=== Model Testing Report ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.weights}\n")
        f.write(f"Test Data: {args.data_path}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Overall Metrics ===\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n=== Detailed Classification Report ===\n")
        f.write(metrics_calc.classification_report())
    print(f"Detailed report saved to: {report_file}")
    
    # Save confusion matrix plot
    if args.save_confusion_matrix:
        try:
            # Regular confusion matrix
            fig = metrics_calc.plot_confusion_matrix(normalize=False, figsize=(10, 8))
            cm_file = output_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(cm_file, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {cm_file}")
            
            # Normalized confusion matrix
            fig_norm = metrics_calc.plot_confusion_matrix(normalize=True, figsize=(10, 8))
            cm_norm_file = output_dir / f"confusion_matrix_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig_norm.savefig(cm_norm_file, dpi=300, bbox_inches='tight')
            print(f"Normalized confusion matrix saved to: {cm_norm_file}")
        except Exception as e:
            print(f"Warning: Could not save confusion matrix plots: {e}")
    
    # Save individual predictions if requested
    if args.save_predictions:
        pred_file = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(pred_file, 'w') as f:
            f.write("path,true_label,predicted_label,confidence,correct\n")
            for i, (path, true_label, pred_label) in enumerate(zip(paths, targets, predictions)):
                confidence = probabilities[i, pred_label].item()
                correct = true_label.item() == pred_label.item()
                f.write(f"{path},{true_label.item()},{pred_label.item()},{confidence:.4f},{correct}\n")
        print(f"Individual predictions saved to: {pred_file}")


def main():
    """Main testing function"""
    # Parse arguments
    args = parse_test_arguments()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Model Testing ===")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.weights}")
    print(f"Test Data: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {device}")
    print()
    
    try:
        # Load model and checkpoint
        model = load_model_and_checkpoint(args, device)
        
        # Create test data loader
        print("Loading test dataset...")
        test_loader = create_dataloader(
            data_dir=Path(args.data_path),
            args=args,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle for testing
            is_train=False,  # Use validation transforms
            extensions=args.extensions,
            return_class_mapping=False,
            use_balanced_sampling=False
        )
        
        if test_loader is None:
            raise ValueError("Test data loader is None. Check your data path and format.")
        
        print(f"Test dataset size: {len(test_loader.dataset)}")
        print(f"Number of batches: {len(test_loader)}")
        print()
        
        # Run inference
        predictions, targets, probabilities, paths = run_inference(model, test_loader, device, args)
        
        print(f"\nTotal samples processed: {len(predictions)}")
        print()
        
        # Calculate metrics
        metrics, metrics_calc = calculate_metrics(predictions, targets, probabilities, args.class_names)
        
        # Print results
        print("=== Test Results ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            print(f"AUC: {metrics['auc']:.4f}")
        print()
        
        # Print per-class metrics if available
        per_class_metrics = [(k, v) for k, v in metrics.items() if k.startswith(('precision_class', 'recall_class', 'f1_class'))]
        if per_class_metrics:
            print("=== Per-Class Metrics ===")
            for metric_name, value in per_class_metrics:
                print(f"{metric_name}: {value:.4f}")
            print()
        
        # Save results
        save_results(args, metrics, metrics_calc, predictions, targets, paths, probabilities)
        
        print("=== Testing Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise


if __name__ == '__main__':
    main()