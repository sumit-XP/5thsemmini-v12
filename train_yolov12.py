"""
YOLOv12 Training Script

This script trains a YOLOv12 model using the downloaded weights and configuration.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path

from ultralytics import YOLO
from config import TRAINING_CONFIG as C
import yaml

def main():
    """Main training function for YOLOv12."""
    parser = argparse.ArgumentParser(description="Train YOLOv12")
    parser.add_argument("--epochs", type=int, default=C.epochs, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=C.batch_size, help="Batch size")
    parser.add_argument("--img", type=int, default=C.img_size, help="Image size")
    parser.add_argument("--device", type=str, default=C.device, help="Device (cuda or cpu)")
    parser.add_argument("--data", type=str, default="yolov8_data.yaml", help="Path to data config file")
    args = parser.parse_args()

    # Create output directory
    save_dir = "runs/train_yolov12"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("YOLOv12 Training Configuration")
    print("=" * 70)
    print(f"Model: {C.model_variant}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model
    model_path = f"{C.model_variant}.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Handle dataset configuration for Kaggle/Local switching (reusing logic)
    data_config = args.data
    if C.dataset_root != "dataset-9": 
        print(f"[Info] Detected custom dataset root: {C.dataset_root}")
        with open(args.data, 'r') as f:
            yaml_data = yaml.safe_load(f)
        yaml_data['path'] = C.dataset_root
        data_config = "yolov12_temp.yaml"
        with open(data_config, 'w') as f:
            yaml.dump(yaml_data, f)

    # Train
    print("\nStarting training...")
    try:
        results = model.train(
            data=data_config,
            epochs=args.epochs,
            imgsz=args.img,
            batch=args.batch,
            device=args.device,
            lr0=C.learning_rate,
            momentum=C.momentum,
            weight_decay=C.weight_decay,
            patience=C.patience,
            save=True,
            project="runs",
            name="train_yolov12",
            exist_ok=True,
            pretrained=True,
            verbose=True
        )
        print("Training completed.")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
