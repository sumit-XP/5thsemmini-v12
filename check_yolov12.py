from ultralytics import YOLO
import os

model_path = "yolov12n.pt"
if not os.path.exists(model_path):
    print(f"Error: {model_path} does not exist.")
else:
    try:
        model = YOLO(model_path)
        print("YOLOv12 loaded successfully from local file!")
    except Exception as e:
        print(f"Failed to load YOLOv12: {e}")
