# Tip Detector - YOLOv8 Implementation

A YOLOv8-based tip detection system for laboratory tip racks.

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
python quick_start.py
```

This will:
1. Install all required dependencies
2. Prepare your labeled data
3. Train a YOLOv8 model
4. Run inference on your training images

### Option 2: Manual Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare data:**
```bash
python prepare_data.py
```

3. **Train the model:**
```bash
python train_yolo.py

export KMP_DUPLICATE_LIB_OK=TRUE && python train_yolo.py
```

4. **Run inference:**
```bash
# Interactive selection - choose which images to test
python inference.py

# Test on a specific image
python inference.py path/to/your/image.jpg
```

## Project Structure

```
tipDetector/
├── data/                    # YOLO dataset structure
│   ├── images/
│   │   ├── train/          # Training images
│   │   ├── val/           # Validation images
│   │   └── test/          # Test images
│   └── labels/
│       ├── train/         # Training labels
│       ├── val/           # Validation labels
│       └── test/          # Test labels
├── yolo_labels/           # Your original labeled data
├── train_jpg/            # Your training images
├── runs/detect/          # Training outputs and model weights
├── inference_results/    # Inference results with bounding boxes
├── dataset.yaml          # YOLO dataset configuration
└── *.py                  # Training and inference scripts
```

## Files Description

- `quick_start.py` - Automated setup and training
- `prepare_data.py` - Data preparation and organization
- `train_yolo.py` - YOLOv8 training script
- `inference.py` - Inference and testing script
- `config.py` - Configuration parameters
- `requirements.txt` - Python dependencies


## Auto-labeling for Label Studio

### Step 1: Generate YOLO labels (without drawing on images)
```bash
python auto_label.py --model runs/detect/tip_detection5/weights/best.pt --images new_training --conf 0.25
```

This creates `auto_labels/runX/` with:
- `images/` → symlink to original images (no annotations drawn)
- `labels/` → YOLO format predictions
- `classes.txt` → class names

### Step 2: Convert to Label Studio format
```bash
cd auto_labels/run1  # or whatever run number was created
label-studio-converter import yolo -i . -o labelstudio_json/output.json --out-type predictions --image-root-url '' --image-ext .jpeg
```

### Step 3: Fix image paths
```bash
python3 -c "import json; data = json.load(open('labelstudio_json/output.json')); [task.update({'data': {'image': task['data']['image'].lstrip('/')}}) for task in data]; json.dump(data, open('labelstudio_json/output.json', 'w'), indent=2)"
```

### Step 4: Import to Label Studio
1. Set local storage to absolute path: `/Users/markqiao/Downloads/KirkLab/tipDetector/auto_labels/run1/images`
2. Import `labelstudio_json/output.json`
3. Predictions will be editable!