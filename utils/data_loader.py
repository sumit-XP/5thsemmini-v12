from __future__ import annotations
from typing import List, Tuple, Dict, Any
import os
import glob
import random
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .augmentation import hybrid_augmentation


class YOLODataset(Dataset):
    """YOLO-format dataset loader.

    Expects structure:
        root/
          images/train/*.jpg
          images/test/*.jpg
          labels/train/*.txt
          labels/test/*.txt

    Labels per line: class x y w h (normalized)
    """

    def __init__(self, root: str, split: str = "train", img_size: int = 640, augment: bool = True, use_mosaic: bool = True) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.use_mosaic = use_mosaic
        self.image_paths = sorted(glob.glob(os.path.join(root, "images", split, "*")))
        self.label_dir = os.path.join(root, "labels", split)

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_image_label(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.image_paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, name + ".txt")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        # Resize to square
        image = cv2.resize(image, (self.img_size, self.img_size))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = parts
                    boxes.append([float(xc), float(yc), float(bw), float(bh)])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        return image, boxes

    def rand_index(self) -> int:
        return random.randint(0, len(self.image_paths) - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, boxes = self.load_image_label(idx)
        if self.augment:
            image, boxes = hybrid_augmentation(image, boxes, self, self.use_mosaic)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
        targets = torch.from_numpy(boxes) if boxes.size else torch.zeros((0, 4), dtype=torch.float32)
        return image, targets


def yolo_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def _worker_init_fn(worker_id: int):
    """Worker init to avoid OpenCV multi-threading conflicts inside DataLoader workers (Windows)."""
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.set_num_threads(1)
    except Exception:
        pass


class COCODataset(Dataset):
    """COCO-format dataset loader for Roboflow exports.

    Expects structure:
        root/
          train/_annotations.coco.json and images in same dir
          valid/_annotations.coco.json
          test/_annotations.coco.json
    """

    def __init__(self, root: str, split: str = "train", img_size: int = 640, augment: bool = True, use_mosaic: bool = True) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.use_mosaic = use_mosaic
        split_dir = os.path.join(root, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(ann_path)
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Build index
        self.images: List[Dict[str, Any]] = data.get("images", [])
        anns: List[Dict[str, Any]] = data.get("annotations", [])
        self.id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns:
            self.id_to_anns.setdefault(a["image_id"], []).append(a)
        self.split_dir = split_dir

    def __len__(self) -> int:
        return len(self.images)

    def rand_index(self) -> int:
        return random.randint(0, len(self.images) - 1)

    def load_image_label(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        info = self.images[idx]
        file_name = info["file_name"]
        img_path = os.path.join(self.split_dir, file_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
        h0, w0 = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        anns = self.id_to_anns.get(info["id"], [])
        boxes: List[List[float]] = []
        for a in anns:
            # COCO bbox: [x,y,w,h] absolute
            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            # convert to normalized xywh
            xc = (x + w / 2.0) / max(1, w0)
            yc = (y + h / 2.0) / max(1, h0)
            bw = w / max(1, w0)
            bh = h / max(1, h0)
            boxes.append([xc, yc, bw, bh])
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        return image, boxes_np

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, boxes = self.load_image_label(idx)
        if self.augment:
            image, boxes = hybrid_augmentation(image, boxes, self, self.use_mosaic)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        targets = torch.from_numpy(boxes) if boxes.size else torch.zeros((0, 4), dtype=torch.float32)
        return image, targets


def create_dataloader(root: str, split: str, img_size: int, batch_size: int, num_workers: int = 2, pin_memory: bool = True, use_mosaic: bool = True):
    yolo_images_dir = os.path.join(root, "images", split)
    coco_split_dir = os.path.join(root, split)
    if os.path.isdir(yolo_images_dir):
        ds = YOLODataset(root, split=split, img_size=img_size, augment=(split == "train"), use_mosaic=use_mosaic)
    elif os.path.isdir(coco_split_dir) and os.path.exists(os.path.join(coco_split_dir, "_annotations.coco.json")):
        ds = COCODataset(root, split=split, img_size=img_size, augment=(split == "train"), use_mosaic=use_mosaic)
    else:
        raise FileNotFoundError(f"Could not find YOLO or COCO split under {root} for split '{split}'")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolo_collate,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0 and split == "train"),
        prefetch_factor=2 if num_workers > 0 else None,  # ignored if num_workers=0
        timeout=120,
        multiprocessing_context="spawn",
    )
    return ds, dl
