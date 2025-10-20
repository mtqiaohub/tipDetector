"""
Configuration file for tip detection training.
Modify these parameters to customize your training.
"""

# Model configuration
MODEL_SIZE = 'n'  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
PRETRAINED_WEIGHTS = 'yolov8n.pt'  # Will be downloaded automatically

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.01
PATIENCE = 20  # Early stopping patience

# Data augmentation
AUGMENTATION = {
    'hsv_h': 0.015,      # HSV-Hue augmentation
    'hsv_s': 0.7,        # HSV-Saturation augmentation  
    'hsv_v': 0.4,        # HSV-Value augmentation
    'degrees': 0.0,      # Image rotation degrees
    'translate': 0.1,    # Image translation
    'scale': 0.5,        # Image scale
    'shear': 0.0,        # Image shear
    'perspective': 0.0,  # Image perspective
    'flipud': 0.0,       # Image flip up-down
    'fliplr': 0.5,       # Image flip left-right
    'mosaic': 1.0,       # Image mosaic
    'mixup': 0.0,        # Image mixup
}

# Inference parameters
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Paths
DATA_DIR = "data"
TRAIN_IMAGES_DIR = "data/images/train"
VAL_IMAGES_DIR = "data/images/val"
TEST_IMAGES_DIR = "data/images/test"
TRAIN_LABELS_DIR = "data/labels/train"
VAL_LABELS_DIR = "data/labels/val"
TEST_LABELS_DIR = "data/labels/test"

# Output directories
OUTPUT_DIR = "runs/detect"
INFERENCE_OUTPUT_DIR = "inference_results"
