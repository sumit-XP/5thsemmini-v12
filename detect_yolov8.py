"""
YOLOv8 Detection/Inference Script for Dairy Cow Behavior Recognition

This script runs inference using a trained YOLOv8 model.
It can process individual images or entire directories.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import glob

from ultralytics import YOLO
from config import TRAINING_CONFIG as C


def main():
    """Main detection function for YOLOv8."""
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference for cow behavior detection")
    parser.add_argument("--source", type=str, default="", help="Image file or directory (uses dataset test split if empty)")
    parser.add_argument("--weights", type=str, default="runs/train_yolov8/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=C.conf_threshold, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=C.iou_threshold, help="NMS IoU threshold")
    parser.add_argument("--img", type=int, default=C.img_size, help="Image size")
    parser.add_argument("--max", type=int, default=5, help="Max number of images to process")
    parser.add_argument("--device", type=str, default=C.device, help="Device (cuda or cpu)")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated images")
    parser.add_argument("--show-labels", action="store_true", default=True, help="Show labels on images")
    parser.add_argument("--show-conf", action="store_true", default=True, help="Show confidence scores")
    args = parser.parse_args()

    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"⚠ Warning: Model weights not found at {args.weights}")
        print("Using a pretrained YOLOv8n model instead for demo purposes.")
        print("To use your trained model, please train first using train_yolov8.py")
        args.weights = "yolov8n.pt"

    print("=" * 70)
    print("YOLOv8 Detection Configuration")
    print("=" * 70)
    print(f"Model weights: {args.weights}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model
    print(f"\n[1/3] Loading model from {args.weights}...")
    model = YOLO(args.weights)
    print("✓ Model loaded successfully")

    # Resolve source paths
    print("\n[2/3] Resolving image sources...")
    paths = []
    if args.source:
        if os.path.isdir(args.source):
            # Directory provided
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for ext in exts:
                paths.extend(sorted(glob.glob(os.path.join(args.source, ext))))
        else:
            # Single file provided
            paths = [args.source]
    else:
        # Use dataset test split
        coco_dir = os.path.join(C.dataset_root, C.test_split)
        yolo_standard_dir = os.path.join(C.dataset_root, "images", C.test_split)
        yolo_dataset9_dir = os.path.join(C.dataset_root, C.test_split, "images")
        
        # Check which format exists
        if os.path.exists(os.path.join(coco_dir, "_annotations.coco.json")):
            sample_dir = coco_dir
        elif os.path.isdir(yolo_dataset9_dir):
            sample_dir = yolo_dataset9_dir
        elif os.path.isdir(yolo_standard_dir):
            sample_dir = yolo_standard_dir
        elif os.path.isdir(coco_dir):
            sample_dir = coco_dir
        else:
            print(f"⚠ Error: Could not find test split at {coco_dir}, {yolo_standard_dir}, or {yolo_dataset9_dir}")
            return
        
        # Get image files
        paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        paths.sort()

    # Limit number of images
    if args.max and len(paths) > args.max:
        paths = paths[:args.max]

    print(f"✓ Found {len(paths)} images to process")

    if not paths:
        print("⚠ No images found to process!")
        return

    # Run inference
    print("\n[3/3] Running inference...")
    print("-" * 70)
    
    output_dir = "runs/detect_yolov8"
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(paths, 1):
        print(f"\nProcessing [{i}/{len(paths)}]: {os.path.basename(img_path)}")
        
        # Run prediction
        results = model.predict(
            source=img_path,
            imgsz=args.img,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=args.save,
            project="runs",
            name="detect_yolov8",
            exist_ok=True,
            show_labels=args.show_labels,
            show_conf=args.show_conf,
            verbose=False,
        )

        # Print detections
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                print(f"  Detected {len(boxes)} objects:")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = C.class_names[cls_id] if cls_id < len(C.class_names) else f"class_{cls_id}"
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(f"    - {class_name}: {conf:.2f} @ ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
            else:
                print("  No objects detected")

    print("\n" + "-" * 70)
    print("✓ Detection complete!")
    print(f"  Annotated images saved to: {output_dir}/")
    print("\nDetailed detection results and visualizations are available in the output directory.")


if __name__ == "__main__":
    main()
