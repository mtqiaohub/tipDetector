#!/usr/bin/env python3
"""
Quick start script for tip detection training and testing.
This script will prepare data, train a model, and run inference.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def main():
    """Main function to run the complete pipeline."""
    print("=== YOLOv8 Tip Detection Quick Start ===\n")
    
    # Step 1: Install requirements
    print("Step 1: Installing requirements...")
    if not install_requirements():
        print("Failed to install requirements. Please install manually.")
        return
    
    # Step 2: Prepare data
    print("\nStep 2: Preparing data...")
    try:
        from prepare_data import prepare_yolo_data
        prepare_yolo_data()
        print("Data preparation completed!")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    # Step 3: Train model
    print("\nStep 3: Training YOLOv8 model...")
    print("This may take a while depending on your hardware...")
    try:
        from train_yolo import train_tip_detector, validate_model
        results = train_tip_detector()
        print("Training completed!")
        
        # Validate the model
        print("\nStep 4: Validating model...")
        validate_model()
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Step 5: Run inference
    print("\nStep 5: Running inference on training images...")
    try:
        from inference import batch_inference_on_training_images
        batch_inference_on_training_images()
        print("Inference completed!")
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    print("\n=== Quick Start Completed! ===")
    print("\nNext steps:")
    print("1. Check the 'inference_results/' folder for detection results")
    print("2. Look at 'runs/detect/tip_detection/' for training metrics")
    print("3. Use 'python inference.py <image_path>' to test on specific images")
    print("4. Use 'python inference.py' to test on all training images")

if __name__ == "__main__":
    main()
