# Dairy Cow Behavior Recognition - YOLOv8 & YOLOv3

A deep learning system for recognizing dairy cow behaviors (standing, lying, walking, mounting). Now upgraded to support **YOLOv8** (recommended) alongside the original custom GC-Res2 YOLOv3 implementation.

## What to do (step-by-step)

- **1) Get the dataset**
  - Put your data at: `c:\\Users\\marri\\Downloads\\mini\\dataset-9` (default in config).
  - Supported layouts:
    - Roboflow COCO: `root/{train,valid,test}/_annotations.coco.json` with images in the same folders.
    - YOLO format: `root/images/{train,valid,test}/*.jpg` and `root/labels/{train,valid,test}/*.txt`.

- **2) Install dependencies (Windows, Python 3.8+)**
  - Open a terminal in the workspace root (folder that contains `project/`).
  - Run:
    ```bat
    python -m pip install -r project\requirements.txt
    ```

- **3) Edit configuration (optional)**
  - Open `project/config.py` and adjust:
    - `dataset_root` if your dataset path differs.
    - `epochs`, `batch_size`, `learning_rate` per your hardware.
    - `device` auto-detects CUDA if available.

- **4) Run tests (quick verification)**
  - From the workspace root:
    ```bat
    python -m pytest -q project\tests
    ```

- **5) Train the model**
  - For a quick smoke run (optional): set `epochs=1` in `config.py`.
  - Then run:
    ```bat
    python project\train.py
    ```
  - Checkpoints will be saved in `runs\train\` (e.g., `final.pth`).

- **6) Run inference on sample images**
  - This script loads `runs\train\final.pth` if present (otherwise runs with random weights).
  - It auto-detects COCO vs YOLO folder layout under your dataset root.
  - Run:
    ```bat
    python project\detect.py
    ```
  - Annotated images are written to `runs\detect\`.

## YOLOv8 Usage (Recommended)

### Why YOLOv8?
- **Better accuracy**: Modern architecture with improved performance
- **Faster training**: Efficient training with built-in optimizations  
- **Easy to use**: High-level API from Ultralytics
- **Pretrained weights**: Transfer learning from COCO dataset

### Quick Start with YOLOv8

**1) Install dependencies**
```bat
python -m pip install -r project\requirements.txt
```

**2) Train YOLOv8**
```bat
python project\train_yolov8.py --epochs 150 --batch 32
```

For a quick smoke test (1 epoch):
```bat
python project\train_yolov8.py --epochs 1 --batch 8
```

**3) Run inference**
```bat
python project\detect_yolov8.py --source dataset-9\test --max 5
```

Or use your own images:
```bat
python project\detect_yolov8.py --source path\to\your\images
```

### YOLOv8 Model Variants
Choose based on your needs (edit `config.py`):
- `yolov8n`: Fastest, smallest (recommended for testing)
- `yolov8s`: Small, good balance
- `yolov8m`: Medium accuracy and speed **(default)**
- `yolov8l`: Large, higher accuracy
- `yolov8x`: Extra large, best accuracy

### YOLOv8 Output
- Trained model: `runs/train_yolov8/weights/best.pt`
- Training plots: `runs/train_yolov8/` (confusion matrix, metrics, etc.)
- Detections: `runs/detect_yolov8/`

## Notes and tips

- **CUDA/AMP**: AMP and cuDNN benchmark are enabled automatically on CUDA. On CPU, AMP is disabled.
- **Augmentations**: Horizontal flip, HSV brightness, and mosaic are applied during training.
- **Loss**: A minimal CIoU-based loss is implemented to support training/tests. For best accuracy, integrate full YOLOv3 target assignment and loss.
- **Common issues**:
  - Install errors: upgrade pip (`python -m pip install -U pip`) and retry.
  - PyTorch/CUDA mismatch: install a CUDA-compatible torch (see pytorch.org).
  - Dataset not found: verify `dataset_root` and folder structure.

## Project layout

- `project/models/`: Res2 backbone, GC blocks, YOLOv3 heads, integrated model.
- `project/utils/`: augmentation, data loader (COCO+YOLO), anchors, loss, visualization, debug utils.
- `project/tests/`: unit and integration tests.
- `project/train.py`: training entrypoint.
- `project/detect.py`: basic inference entrypoint.




# for faster training
- decreased epoch from 200 to 25 ( will lead to less convergance)
- decreased image size from 640 to 300 ( won't affect the accuracy much)
- disabled mosaic augmentation
- have reduced learning rate from 0.1 to 0.005 ( will make the model more stable)
- gradient_accumulation_steps: 2 from 