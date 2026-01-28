#!/usr/bin/env python3
"""
Auto-labeling script that generates YOLO labels for Label Studio without drawing on images.
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

def auto_label(model_path, images_dir, output_dir, conf_threshold=0.25):
    """
    Generate YOLO format labels from model predictions without modifying images.
    
    Args:
        model_path: Path to trained YOLO model
        images_dir: Directory containing images to label
        output_dir: Directory to save labels (will create run1, run2, etc.)
        conf_threshold: Confidence threshold for predictions
    """
    
    # Load model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Create output directory structure
    output_dir = Path(output_dir)
    
    # Find next run number
    run_num = 1
    while (output_dir / f"run{run_num}").exists():
        run_num += 1
    
    run_dir = output_dir / f"run{run_num}"
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlink to original images
    images_symlink = run_dir / "images"
    if images_symlink.exists():
        images_symlink.unlink()
    images_symlink.symlink_to(Path(images_dir).resolve())
    
    print(f"Output directory: {run_dir}")
    print(f"Images: {images_symlink} -> {images_dir}")
    print(f"Labels: {labels_dir}")
    
    # Get image paths
    image_paths = list(Path(images_dir).glob("*.jpeg")) + list(Path(images_dir).glob("*.jpg"))
    
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
    
    print(f"\nProcessing {len(image_paths)} images with confidence threshold {conf_threshold}")
    
    # Process each image
    for img_path in image_paths:
        print(f"  {img_path.name}...", end=" ")
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        r = results[0]
        
        # Save YOLO format labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            if r.boxes is not None and len(r.boxes) > 0:
                # Get image dimensions
                img_h, img_w = r.orig_shape
                
                for box in r.boxes:
                    # Get box coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to YOLO format (class x_center y_center width height, all normalized)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # Get class and confidence
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # Write YOLO format: class x_center y_center width height confidence
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
                
                print(f"{len(r.boxes)} detections")
            else:
                print("no detections")
    
    # Copy classes.txt
    classes_path = labels_dir / "classes.txt"
    with open(classes_path, 'w') as f:
        f.write("rack\ntip\n")
    
    # Copy to root for label-studio-converter
    shutil.copy(classes_path, run_dir / "classes.txt")
    
    print(f"\n✅ Auto-labeling complete!")
    print(f"📁 Labels saved to: {run_dir}")
    print(f"\nNext steps:")
    print(f"1. Review labels in Label Studio")
    print(f"2. Run label-studio-converter:")
    print(f"   cd {run_dir}")
    print(f"   label-studio-converter import yolo -i . -o labelstudio_json/output.json \\")
    print(f"     --out-type predictions --image-root-url '' --image-ext .jpeg")
    print(f"3. Fix paths: python3 -c \"import json; data = json.load(open('labelstudio_json/output.json')); ")
    print(f"   [task.update({{'data': {{'image': task['data']['image'].lstrip('/')}}}}) for task in data]; ")
    print(f"   json.dump(data, open('labelstudio_json/output.json', 'w'), indent=2)\"")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-label images using YOLO model")
    parser.add_argument("--model", default="runs/detect/tip_detection5/weights/best.pt",
                        help="Path to trained YOLO model")
    parser.add_argument("--images", default="new_training",
                        help="Directory containing images to label")
    parser.add_argument("--output", default="auto_labels",
                        help="Output directory for labels")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    
    args = parser.parse_args()
    
    auto_label(args.model, args.images, args.output, args.conf)


