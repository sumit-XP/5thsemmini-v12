from __future__ import annotations
from typing import List, Tuple, Dict
import os
import argparse
import glob
import math

import numpy as np
import torch

from project.config import TRAINING_CONFIG as C
from project.detect import detect_cow_behavior


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def load_yolo_labels(img_path: str, dataset_root: str) -> Tuple[List[int], List[np.ndarray]]:
    # Support YOLO layout: dataset_root/images/<split>/<name>.jpg and labels/<split>/<name>.txt
    # Derive label file path
    rel = None
    if os.path.sep + "images" + os.path.sep in img_path:
        rel = img_path.split(os.path.sep + "images" + os.path.sep, 1)[1]
        label_path = os.path.join(dataset_root, "labels", rel)
        label_path = os.path.splitext(label_path)[0] + ".txt"
    else:
        # Fallback: assume sibling 'labels' next to image file
        base, name = os.path.split(img_path)
        parent = os.path.dirname(base)
        split = os.path.basename(base)
        label_path = os.path.join(parent, "labels", split, os.path.splitext(name)[0] + ".txt")

    cls_ids: List[int] = []
    boxes: List[np.ndarray] = []
    if not os.path.exists(label_path):
        return cls_ids, boxes
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                # Convert normalized xywh to absolute xyxy using current C.img_size baseline is wrong; must map to original image size.
                # Since detect resizes to C.img_size, we compare in pixel space of the actual image by reading size from detect outputs.
                # For evaluation, we'll load the actual image size using cv2 when needed in detect_cow_behavior; here we can't.
                # Instead, we will compute absolute coords later after we know H_img, W_img via a small helper.
                boxes.append(np.array([x, y, w, h], dtype=np.float32))
                cls_ids.append(cid)
    except Exception:
        pass
    return cls_ids, boxes


def xywhn_to_xyxy(xywhn: np.ndarray, W: int, H: int) -> np.ndarray:
    x, y, w, h = xywhn
    cx = x * W
    cy = y * H
    bw = w * W
    bh = h * H
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(W - 1.0, cx + bw / 2.0)
    y2 = min(H - 1.0, cy + bh / 2.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def precision_recall_ap(tp: np.ndarray, fp: np.ndarray, npos: int) -> Tuple[float, float, float]:
    # Sort by score already handled before constructing tp/fp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall = tp_cum / max(npos, 1)
    # AP via 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        p = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap += p / 101.0
    # F1 at best threshold (max over points)
    f1 = 0.0
    if precision.size:
        f1 = np.max(2 * precision * recall / np.maximum(precision + recall, 1e-9))
    return float(precision[-1] if precision.size else 0.0), float(recall[-1] if recall.size else 0.0), float(ap)


def evaluate(split: str, conf: float, iou_thr: float, max_imgs: int) -> None:
    # Resolve image list from dataset
    coco_dir = os.path.join(C.dataset_root, split)
    yolo_dir = os.path.join(C.dataset_root, "images", split)
    img_dir = coco_dir if os.path.exists(os.path.join(coco_dir, "_annotations.coco.json")) else yolo_dir
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images directory not found for split '{split}': {img_dir}")
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(sorted(glob.glob(os.path.join(img_dir, ext))))
    if max_imgs and len(paths) > max_imgs:
        paths = paths[:max_imgs]

    # Build model once
    from project.models.gc_res2_yolov3 import GCRes2YOLOv3
    import cv2
    model = GCRes2YOLOv3(num_classes=C.num_classes).to(C.device)
    weights = os.path.join(C.save_dir, "final.pth")
    if os.path.exists(weights):
        ckpt = torch.load(weights, map_location=C.device)
        model.load_state_dict(ckpt["model"])  # type: ignore

    # Accumulators per class
    det_by_class: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(C.num_classes)}  # (score, tp) per det
    npos_by_class: Dict[int, int] = {i: 0 for i in range(C.num_classes)}

    for img_path in paths:
        # Load GT boxes
        cls_ids_gt, boxes_xywhn = load_yolo_labels(img_path, C.dataset_root)
        # Read image size
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        gt_boxes_xyxy = [xywhn_to_xyxy(b, W, H) for b in boxes_xywhn]
        used = [False] * len(gt_boxes_xyxy)
        for cid in cls_ids_gt:
            if 0 <= cid < C.num_classes:
                npos_by_class[cid] += 1
        
        # Run detection
        annotated, dets = detect_cow_behavior(img_path, model, conf_threshold=conf, iou_threshold=iou_thr)
        # Sort detections by score descending
        dets_sorted = sorted(dets, key=lambda x: x[1], reverse=True)
        
        for name, score, (x1, y1, x2, y2) in dets_sorted:
            try:
                cls_id = C.class_names.index(name)
            except ValueError:
                cls_id = 0
            box_det = np.array([x1, y1, x2, y2], dtype=np.float32)
            # Match to best GT of same class
            best_iou = 0.0
            best_j = -1
            for j, (gt_box, gt_cid) in enumerate(zip(gt_boxes_xyxy, cls_ids_gt)):
                if used[j]:
                    continue
                if gt_cid != cls_id:
                    continue
                iou = iou_xyxy(box_det, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            tp = 1 if best_iou >= iou_thr and best_j >= 0 else 0
            if tp:
                used[best_j] = True
            det_by_class.setdefault(cls_id, []).append((score, tp))

    # Compute metrics per class
    print("\nEvaluation results (split=%s, conf=%.2f, IoU=%.2f):" % (split, conf, iou_thr))
    aps = []
    for cid in range(C.num_classes):
        entries = det_by_class.get(cid, [])
        if entries:
            entries.sort(key=lambda x: x[0], reverse=True)
            scores, tps = zip(*entries)
            tp_arr = np.array(tps, dtype=np.float32)
            fp_arr = 1.0 - tp_arr
            P, R, AP = precision_recall_ap(tp_arr, fp_arr, npos_by_class.get(cid, 0))
        else:
            P, R, AP = 0.0, 0.0, 0.0
        aps.append(AP)
        name = C.class_names[cid] if cid < len(C.class_names) else str(cid)
        print(f"- {name:>12s} | P {P:.3f} R {R:.3f} AP@0.5 {AP:.3f} (GT {npos_by_class.get(cid,0)})")
    mAP = float(np.mean(aps)) if aps else 0.0
    print(f"\n=> mAP@0.5: {mAP:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on YOLO-format dataset (demo decoder)")
    parser.add_argument("--split", type=str, default=C.test_split, help="Dataset split: train|test|val")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP")
    parser.add_argument("--max", type=int, default=500, help="Max images")
    args = parser.parse_args()

    evaluate(split=args.split, conf=args.conf, iou_thr=args.iou, max_imgs=args.max)


if __name__ == "__main__":
    main()
