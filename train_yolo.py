#!/usr/bin/env python3
"""
YOLOv8 training script for tip detection.
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def train_tip_detector():
    """Train YOLOv8 model for tip detection."""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLOv8 model (using nano version for faster training)
    model = YOLO('yolov8n.pt')  # nano version for quick training
    
    # Training parameters
    training_args = {
        'data': 'dataset.yaml',
        'epochs': 100,  # Start with 100 epochs
        'imgsz': 640,   # Image size
        'batch': 16,    # Batch size (adjust based on your GPU memory)
        'device': device,
        'project': 'runs/detect',
        'name': 'tip_detection',
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': 20,     # Early stopping patience
        'lr0': 0.01,        # Initial learning rate
        'lrf': 0.01,        # Final learning rate
        'momentum': 0.937,   # SGD momentum
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3,     # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,   # Warmup initial bias lr
        'box': 7.5,             # Box loss gain
        'cls': 0.5,             # Class loss gain
        'dfl': 1.5,             # DFL loss gain
        'pose': 12.0,           # Pose loss gain
        'kobj': 2.0,            # Keypoint obj loss gain
        'label_smoothing': 0.0, # Label smoothing
        'nbs': 64,              # Nominal batch size
        'hsv_h': 0.015,         # Image HSV-Hue augmentation
        'hsv_s': 0.7,           # Image HSV-Saturation augmentation
        'hsv_v': 0.4,           # Image HSV-Value augmentation
        'degrees': 0.0,         # Image rotation degrees
        'translate': 0.1,       # Image translation
        'scale': 0.5,           # Image scale
        'shear': 0.0,           # Image shear
        'perspective': 0.0,     # Image perspective
        'flipud': 0.0,          # Image flip up-down
        'fliplr': 0.5,          # Image flip left-right
        'mosaic': 1.0,          # Image mosaic
        'mixup': 0.0,           # Image mixup
        'copy_paste': 0.0,      # Segment copy-paste
    }
    
    print("Starting YOLOv8 training...")
    print(f"Training arguments: {training_args}")
    
    # Start training
    results = model.train(**training_args)
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    return results

def validate_model(model_path=None):
    """Validate the trained model."""
    if model_path is None:
        # Find the best model from training
        runs_dir = Path("runs/detect/tip_detection")
        if runs_dir.exists():
            best_model = runs_dir / "weights" / "best.pt"
            if best_model.exists():
                model_path = str(best_model)
            else:
                print("No trained model found!")
                return
        else:
            print("No training runs found!")
            return
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate on validation set
    val_results = model.val(data='dataset.yaml')
    
    print("Validation results:")
    print(f"mAP50: {val_results.box.map50:.3f}")
    print(f"mAP50-95: {val_results.box.map:.3f}")
    
    return val_results

if __name__ == "__main__":
    # First prepare the data
    print("Preparing data...")
    from prepare_data import prepare_yolo_data
    prepare_yolo_data()
    
    # Train the model
    print("\nStarting training...")
    results = train_tip_detector()
    
    # Validate the model
    print("\nValidating model...")
    validate_model()
