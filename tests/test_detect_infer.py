import os
import tempfile
import numpy as np
import cv2
import torch

from project.detect import detect_cow_behavior
from project.models.gc_res2_yolov3 import GCRes2YOLOv3
from project.config import TRAINING_CONFIG as C


def test_detect_runs_on_dummy_image():
    h, w = C.img_size, C.img_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dummy.jpg")
        cv2.imwrite(path, img)
        model = GCRes2YOLOv3(num_classes=C.num_classes).to(C.device)
        annotated, dets = detect_cow_behavior(path, model, C.conf_threshold)
        assert isinstance(dets, list)
        assert annotated.shape == img.shape
