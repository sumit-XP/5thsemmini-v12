"""
YOLOv12 Detection Script
"""
from __future__ import annotations
import os
import argparse
import glob
from ultralytics import YOLO
from config import TRAINING_CONFIG as C

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv12 inference")
    parser.add_argument("--source", type=str, default="", help="Image source")
    parser.add_argument("--weights", type=str, default="runs/train_yolov12/weights/best.pt", help="Weights path")
    parser.add_argument("--conf", type=float, default=C.conf_threshold, help="Confidence threshold")
    parser.add_argument("--max", type=int, default=5, help="Max images")
    parser.add_argument("--device", type=str, default=C.device, help="Device")
    args = parser.parse_args()

    # Fallback to base model if trained weights don't exist
    if not os.path.exists(args.weights):
        print(f"Weights {args.weights} not found, using base model {C.model_variant}.pt")
        args.weights = f"{C.model_variant}.pt"

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Resolve source
    paths = []
    if args.source:
        paths = [args.source] # Simplified for now
    else:
         # Use dataset test split
        yolo_dataset9_dir = os.path.join(C.dataset_root, C.test_split, "images")
        if os.path.isdir(yolo_dataset9_dir):
             paths = [os.path.join(yolo_dataset9_dir, f) for f in os.listdir(yolo_dataset9_dir) 
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if args.max:
        paths = paths[:args.max]

    print(f"Processing {len(paths)} images...")
    for img in paths:
        model.predict(img, conf=args.conf, save=True, project="runs", name="detect_yolov12", exist_ok=True)

if __name__ == "__main__":
    main()
