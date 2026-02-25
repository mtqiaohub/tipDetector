#!/usr/bin/env python3
"""
Inference script for tip detection using trained YOLOv8 model.
"""
import csv
import io
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
# import matplotlib.pyplot as plt

def run_live_inference(model_path, conf_threshold=0.25, interval_s=0.5):
    """
    Smooth video; run YOLO inference ~every `interval_s` seconds.
    Draw last-known detections on every frame in between.
    """
    import time

    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Could not access webcam. Check permissions or camera availability.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    last_result = None
    last_infer_time = 0.0


    frozen = False
    freeze_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from webcam.")
            break

        now = time.time()

        if not frozen:
   
            if (now - last_infer_time) >= interval_s:
                results = model(frame, conf=conf_threshold)
                r = results[0]
                last_result = r
                last_infer_time = now

                # Optional: console debug
                if r.boxes is None or len(r.boxes) == 0:
                    print("⚠️ No detections in this frame.")
                else:
                    print(f"✅ Detected {len(r.boxes)} objects.")
                    # print(r.boxes.data.cpu().numpy())  # uncomment if you want raw numbers

            # Draw overlay using the most recent detections (if any)
            if last_result is not None and last_result.boxes is not None and len(last_result.boxes) > 0:

                boxes_np = last_result.boxes.data.cpu().numpy()
                rack_boxes = boxes_np[boxes_np[:, 5] == 0]
                if len(rack_boxes) > 1:
                    best_rack_idx = np.argmax(rack_boxes[:, 4])
                    rack_indices = np.where(last_result.boxes.cls.cpu().numpy() == 0)[0]
                    keep_rack_idx = rack_indices[best_rack_idx]
                    tip_indices = np.where(last_result.boxes.cls.cpu().numpy() == 1)[0]
                    keep_indices = np.concatenate(([keep_rack_idx], tip_indices))
                    last_result.boxes = last_result.boxes[keep_indices]

                annotated_frame = create_custom_visualization(
                    last_result, conf_threshold, base_frame=frame
                )
                # Small age indicator so you know how “stale” detections are
                age = now - last_infer_time
                cv2.putText(
                    annotated_frame,
                    f"det age: {age:.1f}s",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    f"det age: {age:.1f}s",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                annotated_frame = frame
        
            display_frame = annotated_frame
        else:
            display_frame = freeze_frame
        
        cv2.imshow("Live Tip Detection", display_frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key & 0xFF == ord('c'):

            if frozen:
                frozen = False
            else:
                freeze_frame = frame.copy()
                frozen = True
                visualize_detections(freeze_frame, model=model, conf_threshold=conf_threshold, from_frame=True)


    cap.release()
    cv2.destroyAllWindows()



def run_inference(model_path, image_path, conf_threshold=0.1, save_results=True):
    """
    Run inference on a single image or directory of images.
    
    Args:
        model_path: Path to the trained YOLO model
        image_path: Path to image or directory of images
        conf_threshold: Confidence threshold for detections
        save_results: Whether to save results with bounding boxes
    """
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Get image paths
    if Path(image_path).is_file():
        image_paths = [image_path]
    else:
        image_paths = list(Path(image_path).glob("*.jpeg")) + list(Path(image_path).glob("*.jpg"))
    
    print(f"Running inference on {len(image_paths)} images...")
    print(f"Confidence threshold: {conf_threshold}")
    
    results = []
    
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        
        # Run inference
        result = model(str(img_path), conf=conf_threshold)
        
        # Get the first result (since we're processing one image at a time)
        r = result[0]
        
        # Print detection info
        if r.boxes is not None and len(r.boxes) > 0:
            print(f"  Found {len(r.boxes)} detections above confidence threshold {conf_threshold}")
            class_names = ['rack', 'tip']
            for i, box in enumerate(r.boxes):
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = class_names[cls] if cls < len(class_names) else f'class{cls}'
                print(f"    {class_name.capitalize()} {i+1}: confidence = {conf:.3f}")
        else:
            print(f"  No detections above confidence threshold {conf_threshold}")
        
        # Save results with custom green bounding boxes
        if save_results:
            # Create output directory
            output_dir = Path("inference_results")
            output_dir.mkdir(exist_ok=True)
            
            # Create custom visualization with green boxes
            annotated_img = create_custom_visualization(r, conf_threshold)
            output_path = output_dir / f"result_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
            print(f"  Saved result to: {output_path}")
        
        results.append(r)
    
    return results

def create_custom_visualization(result, conf_threshold=0.1, base_frame=None):
    """
    Create custom visualization with green boxes for tips above confidence threshold.
    """

    if base_frame is not None:
        img = base_frame.copy()
    elif hasattr(result, 'orig_img') and result.orig_img is not None:
        img = result.orig_img.copy()
    elif hasattr(result, 'path') and result.path:
        img = cv2.imread(result.path)
    else:
        raise ValueError("No image available for visualization.")

    # Load the original image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Sort detections by confidence (highest first) to prioritize high-confidence labels
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            conf = box.conf.item()
            if conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls.item())  # Get class ID
                detections.append((conf, x1, y1, x2, y2, i, cls))
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x[0], reverse=True)
    
    # Draw boxes and labels with better positioning
    for conf, x1, y1, x2, y2, idx, cls in detections:
        # Define colors for different classes
        colors = [(255, 0, 0), (0, 255, 0)]  # rack = blue, tip = green
        color = colors[cls] if cls < len(colors) else (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)

        # Create smaller label
        class_names = ['rack', 'tip']
        class_name = class_names[cls] if cls < len(class_names) else f'class{cls}'
        label = f"{class_name} {conf:.2f}"

        # Smaller font, thinner text, tighter padding
        font_scale = 0.6
        font_thickness = 1
        padding = 3

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Position label **above** the rectangle
        label_x = x1
        label_y = max(y1 - 8, text_height + 8)  # ensure it doesn’t go off-screen

        # Draw small solid background box for readability
        cv2.rectangle(
            img_rgb,
            (label_x - padding, label_y - text_height - padding),
            (label_x + text_width + padding, label_y + baseline + padding),
            color,
            -1,
        )

        # Draw text in black for contrast
        cv2.putText(
            img_rgb,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
    
    # Convert back to BGR for saving
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def batch_inference_on_training_images():
    """Run inference on all training images to see how well the model performs."""
    
    # Find the best trained model
    runs_dir = Path("runs/detect/tip_detection5")
    if not runs_dir.exists():
        print("No training runs found! Please train the model first.")
        return
    
    best_model = runs_dir / "weights" / "best.pt"
    if not best_model.exists():
        print("No trained model found! Please train the model first.")
        return
    
    print(f"Using model: {best_model}")
    
    # Run inference on training images
    training_images_dir = "train_jpg"
    if not Path(training_images_dir).exists():
        print(f"Training images directory {training_images_dir} not found!")
        return
    
    results = run_inference(
        model_path=str(best_model),
        image_path=training_images_dir,
        conf_threshold=0.25,
        save_results=True
    )
    
    print(f"\nInference completed! Results saved to inference_results/")
    return results

def calculate_hamming_dist(occupancy):
    total_tips = np.sum(occupancy)
    target_occupancy = np.zeros((8,12), dtype = bool)
    flat_target = target_occupancy.flatten()
    flat_target[-total_tips:] = True
    target_reshaped = flat_target.reshape(8,12)
    hamming_dist = np.sum(target_reshaped != occupancy)
    return hamming_dist, target_reshaped


def tiprack_state_to_csv(occupancy):
    """
    Build a 96-row CSV of the tiprack state: one row per well, two columns (well, filled).

    occupancy: np.ndarray of shape (8, 12), dtype bool. occupancy[row, col] True = tip present.
    Order: A1–A12, B1–B12, … H12 (top to bottom).
    """
    rows, cols = 8, 12
    if occupancy.shape != (rows, cols):
        raise ValueError(f"occupancy must be shape (8, 12), got {occupancy.shape}")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["well", "filled"])
    row_labels = "ABCDEFGH"
    for r in range(rows):
        for c in range(cols):
            well = f"{row_labels[r]}{c + 1}"
            filled = 1 if occupancy[r, c] else 0
            w.writerow([well, filled])
    return buf.getvalue()

def list_transfers(source, target):
    # Identify source-only and target-only coordinates
    source_only = source & ~target
    source_indices = np.argwhere(source_only)

    target_only = target & ~source
    target_indices = np.argwhere(target_only)

    transfers = []

    # Copy arrays to mark progress
    source_used = np.zeros(len(source_indices), dtype=bool)
    target_filled = np.zeros(len(target_indices), dtype=bool)

    for i, t in enumerate(target_indices):

        if target_filled[i]:
            continue

        available_sources = source_indices[~source_used]
        if len(available_sources) == 0:
            break  

        dists = np.linalg.norm(available_sources - t, axis=1)
        closest_idx = np.argmin(dists)

        chosen_source = available_sources[closest_idx]
        transfers.append((tuple(chosen_source), tuple(t)))

        source_used[np.where(~source_used)[0][closest_idx]] = True
        target_filled[i] = True

    return transfers

def visualize_detections(image, model_path=None, model=None, conf_threshold=0.1, from_frame=False):
    """
    Visualize YOLO detections on an image (path or OpenCV frame).
    Can be called from inference.py or live feed.
    """
    import matplotlib.pyplot as plt
    import cv2

    # --- Load model only if not passed in ---
    if model is None:
        if model_path is None:
            raise ValueError("Need either model_path or loaded model.")
        model = YOLO(model_path)

    # --- Handle image input ---
    if from_frame:
        img = image.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Run YOLO inference ---
    results = model(img, conf=conf_threshold)
    r = results[0]

    annotated_img = create_custom_visualization(r, conf_threshold, base_frame=img)
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    boxes = r.boxes.data.cpu().numpy()

    rack_boxes = boxes[boxes[:, 5] == 0]
    tip_boxes  = boxes[boxes[:, 5] == 1]

    if len(rack_boxes) > 1:
        best_rack_idx = np.argmax(rack_boxes[:, 4])
        rack_indices = np.where(r.boxes.cls.cpu().numpy() == 0)[0]
        keep_rack_idx = rack_indices[best_rack_idx]
        tip_indices = np.where(r.boxes.cls.cpu().numpy() == 1)[0]
        keep_indices = np.concatenate(([keep_rack_idx], tip_indices))
        r.boxes = r.boxes[keep_indices]

    boxes = r.boxes.data.cpu().numpy()
    rack_boxes = boxes[boxes[:, 5] == 0]
    tip_boxes  = boxes[boxes[:, 5] == 1]

    if len(rack_boxes) == 0 or len(tip_boxes) == 0:
        print("⚠️ No rack or tips detected.")
        return None

    # --- (same grid/occupancy logic as before) ---
    rack_x1, rack_y1, rack_x2, rack_y2, *_ = rack_boxes[0]
    rack_w, rack_h = rack_x2 - rack_x1, rack_y2 - rack_y1

    rows, cols = 8, 12
    cell_w, cell_h = rack_w / cols, rack_h / rows

    grid_centers = [
        (rack_x1 + (c + 0.5) * cell_w, rack_y1 + (r + 0.5) * cell_h)
        for r in range(rows) for c in range(cols)
    ]

    occupancy = np.zeros((rows, cols), dtype=bool)
    for i, (cx, cy) in enumerate(grid_centers):
        for (x1, y1, x2, y2, _, _) in tip_boxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                r_idx, c_idx = divmod(i, cols)
                occupancy[r_idx, c_idx] = True
                break

    hamming_dist, target = calculate_hamming_dist(occupancy)
    transfers = list_transfers(occupancy, target)

    print("hamming dist:", hamming_dist)
    print("transfers:", transfers)

    print(tiprack_state_to_csv(occupancy))
    # --- 4-panel visualization (same plotting code as before) ---
    visualize_grid_panels(img_rgb, occupancy, target, transfers, conf_threshold, annotated_rgb)
    return occupancy

def visualize_grid_panels(img_rgb, occupancy, target, transfers, conf_threshold, annotated_rgb):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 7))

    # Panel 1: Original
    ax1.imshow(img_rgb)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Panel 2: Placeholder for YOLO boxes (or overlay)
    ax2.imshow(annotated_rgb)
    ax2.set_title(f"Detections (conf ≥ {conf_threshold:.2f})")
    ax2.axis('off')

    # Panel 3: Current occupancy
    ax3.imshow(np.ones_like(occupancy), cmap="gray", vmin=0, vmax=1)
    for r in range(8):
        for c in range(12):
            rr = 7 - r
            color = "limegreen" if occupancy[rr, c] else "lightgray"
            ax3.scatter(c, r, c=color, s=350, marker='o', edgecolor='black')
    ax3.set_title("Current Rack (Occupancy)")
    ax3.invert_yaxis()

    # Panel 4: Planned transfers
    ax4.imshow(np.ones_like(target), cmap="gray", vmin=0, vmax=1)
    for r in range(8):
        for c in range(12):
            rr = 7 - r
            if occupancy[rr, c]:
                ax4.scatter(c, r, c="limegreen", s=350, marker='o', edgecolor='black')
            elif target[rr, c]:
                ax4.scatter(c, r, c="dodgerblue", s=350, marker='o', edgecolor='black')
            else:
                ax4.scatter(c, r, c="lightgray", s=350, marker='o', edgecolor='black')

    for (src, dst) in transfers:
        r1, c1 = 7 - src[0], src[1]
        r2, c2 = 7 - dst[0], dst[1]
        ax4.arrow(
            c1, r1, c2 - c1, r2 - r1,
            color='yellow', width=0.05, head_width=0.3,
            length_includes_head=True, alpha=0.8
        )

    ax4.set_title("Transfers (Source → Target)")
    ax4.invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Run YOLOv8 tip detection inference')
    parser.add_argument('image_path', nargs='?', help='Path to image file or directory')
    parser.add_argument('--conf', '--confidence', type=float, default=0.1, 
                       help='Confidence threshold for detections (default: 0.1)')
    parser.add_argument('--live', action='store_true', help='Run live webcam inference')

    args = parser.parse_args()
    
    model_path = "runs/detect/tip_detection5/weights/best.pt"
    


    
    if not Path(model_path).exists():
        print("No trained model found! Please train the model first.")
        sys.exit(1)
    if args.live:
        run_live_inference(model_path, conf_threshold=args.conf)
    else:
        # Run inference on specific image with specified confidence threshold
        print(f"Running inference on: {args.image_path}")
        print(f"Confidence threshold: {args.conf}")
        
        if Path(args.image_path).is_file():
            # Single image
            visualize_detections(
                image=args.image_path,
                model_path=model_path,
                conf_threshold=args.conf
            )
            
        else:
            # Directory of images
            run_inference(model_path, args.image_path, args.conf, save_results=True)
