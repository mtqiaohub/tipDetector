#!/usr/bin/env python3
"""
Data preparation script for YOLOv8 tip detection.
Organizes labeled data into train/val/test splits and creates YAML config.
"""

import os
import shutil
import random
from pathlib import Path
import yaml

def prepare_yolo_data():
    """Prepare YOLO dataset structure from existing labeled data."""
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Paths
    base_dir = Path(".")
    yolo_labels_dir = base_dir / "yolo_labels"
    train_jpg_dir = base_dir / "train_jpg"
    
    # Create data directories
    data_dirs = [
        "data/images/train", "data/images/val", "data/images/test",
        "data/labels/train", "data/labels/val", "data/labels/test"
    ]
    
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get labeled images from yolo_labels/images
    labeled_images = []
    labeled_img_dir = yolo_labels_dir / "images"
    labeled_lbl_dir = yolo_labels_dir / "labels"
    
    if labeled_img_dir.exists():
        for img_path in labeled_img_dir.glob("*.jpeg"):
            # Look for corresponding label file
            label_name = img_path.stem  # Remove extension
            label_path = labeled_lbl_dir / f"{label_name}.txt"
            
            if label_path.exists():
                labeled_images.append((img_path, label_path))
                print(f"Found labels for: {img_path.name}")
    
    print(f"Found {len(labeled_images)} labeled images")
    
    # Also get unlabeled images from train_jpg for validation
    all_images = list(train_jpg_dir.glob("*.jpeg"))
    print(f"Found {len(all_images)} unlabeled images in train_jpg/")
    
    if len(labeled_images) == 0:
        print("No labeled images found! Please check your yolo_labels folder.")
        return
    
    # Split labeled images between training and validation
    if len(labeled_images) == 1:
        # Single image: use for both training and validation
        img_path, label_path = labeled_images[0]
        
        # Training set
        shutil.copy2(img_path, "data/images/train/")
        shutil.copy2(label_path, "data/labels/train/")
        
        # Validation set (same image)
        shutil.copy2(img_path, "data/images/val/")
        shutil.copy2(label_path, "data/labels/val/")
        
        print("Using the same labeled image for both training and validation")
        print("This is not ideal but necessary with only one labeled image")
        
    elif len(labeled_images) >= 2:
        # Multiple images: split between training and validation
        # Use 2/3 for training, 1/3 for validation
        train_count = max(1, int(len(labeled_images) * 2/3))
        val_count = len(labeled_images) - train_count
        
        # Shuffle the images for random split
        random.shuffle(labeled_images)
        
        # Training set
        for i in range(train_count):
            img_path, label_path = labeled_images[i]
            shutil.copy2(img_path, "data/images/train/")
            shutil.copy2(label_path, "data/labels/train/")
        
        # Validation set
        for i in range(train_count, len(labeled_images)):
            img_path, label_path = labeled_images[i]
            shutil.copy2(img_path, "data/images/val/")
            shutil.copy2(label_path, "data/labels/val/")
        
        print(f"Split {len(labeled_images)} labeled images: {train_count} for training, {val_count} for validation")
        
    else:
        print("No labeled images found! Please check your yolo_labels folder.")
        return
    
    # Create dataset YAML file
    dataset_config = {
        'path': str(base_dir.absolute()),
        'train': 'data/images/train',
        'val': 'data/images/val',
        'test': 'data/images/test',
        'nc': 2,  # number of classes
        'names': ['rack', 'tip']  # class names
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("Dataset structure created!")
    print("Training images:", len(list(Path("data/images/train").glob("*"))))
    print("Validation images:", len(list(Path("data/images/val").glob("*"))))
    print("Training labels:", len(list(Path("data/labels/train").glob("*"))))
    print("Validation labels:", len(list(Path("data/labels/val").glob("*"))))

if __name__ == "__main__":
    prepare_yolo_data()
