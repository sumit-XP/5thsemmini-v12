"""
YOLOv8 Training Script for Dairy Cow Behavior Recognition

This script trains a YOLOv8 model using the Ultralytics library.
It uses the configuration from config.py and the dataset defined in yolov8_data.yaml.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path

from ultralytics import YOLO
from config import TRAINING_CONFIG as C


def main():
    """Main training function for YOLOv8."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 for cow behavior recognition")
    parser.add_argument("--epochs", type=int, default=C.epochs, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=C.batch_size, help="Batch size")
    parser.add_argument("--img", type=int, default=C.img_size, help="Image size")
    parser.add_argument("--model", type=str, default=C.model_variant, help="YOLOv8 model variant (n/s/m/l/x)")
    parser.add_argument("--pretrained", action="store_true", default=C.pretrained, help="Use pretrained weights")
    parser.add_argument("--device", type=str, default=C.device, help="Device (cuda or cpu)")
    parser.add_argument("--data", type=str, default="yolov8_data.yaml", help="Path to data config file")
    args = parser.parse_args()

    # Create output directory
    save_dir = "runs/train_yolov8"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("YOLOv8 Training Configuration")
    print("=" * 70)
    print(f"Model variant: {args.model}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}")
    print(f"Dataset config: {args.data}")
    print(f"Learning rate: {C.learning_rate}")
    print(f"Momentum: {C.momentum}")
    print(f"Weight decay: {C.weight_decay}")
    print(f"Patience: {C.patience}")
    print(f"Output directory: {save_dir}")
    print("=" * 70)

    # Initialize model
    if args.pretrained:
        print(f"\n[1/3] Loading pretrained YOLOv8 model: {args.model}.pt")
        model = YOLO(f"{args.model}.pt")
        print("✓ Pretrained model loaded successfully")
    else:
        print(f"\n[1/3] Creating YOLOv8 model from scratch: {args.model}.yaml")
        model = YOLO(f"{args.model}.yaml")
        print("✓ Model created successfully")

    # Handle dataset configuration for Kaggle/Local switching
    data_config = args.data
    if C.dataset_root != "dataset-9": # Modified by config for Kaggle
        print(f"\n[Info] Detected custom dataset root: {C.dataset_root}")
        
        # Read original yaml
        import yaml
        with open(args.data, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Update path
        yaml_data['path'] = C.dataset_root
        
        # Save to temp file
        data_config = "yolov8_temp.yaml"
        with open(data_config, 'w') as f:
            yaml.dump(yaml_data, f)
        print(f"[Info] Created temporary config '{data_config}' with updated path.")

    # Train the model
    print("\n[2/3] Starting training...")
    print("-" * 70)
    
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
        save_period=C.save_every,
        project="runs",
        name="train_yolov8",
        exist_ok=True,
        pretrained=args.pretrained,
        optimizer="SGD",
        verbose=True,
        # Additional configurations
        workers=C.num_workers,
        amp=C.mixed_precision,
        warmup_epochs=C.warmup_epochs,
        mosaic=1.0 if C.use_mosaic else 0.0,
        close_mosaic=10,  # Disable mosaic in last 10 epochs
    )

    print("-" * 70)
    print("✓ Training completed successfully!")

    # Model is automatically saved by Ultralytics
    model_path = Path("runs/train_yolov8/weights/best.pt")
    print(f"\n[3/3] Model saved to: {model_path}")
    
    # Print training summary
    if hasattr(results, 'results_dict'):
        print("\nTraining Summary:")
        print("-" * 70)
        metrics = results.results_dict
        if 'metrics/mAP50(B)' in metrics:
            print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
        print("-" * 70)

    print("\n✓ All done! You can now run inference with detect_yolov8.py")
    print(f"  Model weights: runs/train_yolov8/weights/best.pt")
    print(f"  Training plots: runs/train_yolov8/")


if __name__ == "__main__":
    main()
