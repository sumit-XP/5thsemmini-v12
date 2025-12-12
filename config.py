from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch




@dataclass
class Config:
    """Configuration parameters for training and inference."""
    # Data - COCO FORMAT (Roboflow)
    dataset_root: str = "dataset-9"
    images_subdir: str = ""  # Empty for COCO format
    labels_subdir: str = ""  # Empty for COCO format
    train_split: str = "train"
    test_split: str = "test"
    img_size: int = 640



    # Model - CORRECTED CLASS COUNT
    num_classes: int = 4  # ✅ CHANGED: 4 classes (Drinking, Eating, Sitting, Standing)
    class_names: List[str] = None
    anchors: Optional[List[Tuple[int, int]]] = None

    # Training - Optimized for Kaggle
    batch_size: int = 16
    epochs: int = 120
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.005
    momentum: float = 0.937
    weight_decay: float = 5e-4
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    warmup_epochs: int = 5
    scheduler: str = "cosine"


    # Performance
    num_workers: int = 1
    pin_memory: bool = True
    mixed_precision: bool = True
    use_compile: bool = False
    use_mosaic: bool = True
    
    # Checkpoints/Logging
    save_dir: str = "runs/train"
    save_every: int = 10



    # Misc
    device: str = "cuda"

    # YOLOv8 specific
    model_variant: str = "yolov12m"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    pretrained: bool = True  # Use pretrained weights
    patience: int = 50  # Early stopping patience



    def __post_init__(self) -> None:
        if self.class_names is None:
            # ✅ CORRECTED: Match actual dataset classes
            self.class_names = ["Drinking", "Eating", "Sitting", "Standing"]




TRAINING_CONFIG = Config()
