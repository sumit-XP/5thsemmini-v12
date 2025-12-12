from __future__ import annotations
from typing import List, Tuple
import os
import argparse
import glob

import cv2
import numpy as np
import torch

from project.config import TRAINING_CONFIG as C
from project.models.gc_res2_yolov3 import GCRes2YOLOv3


@torch.no_grad()
def detect_cow_behavior(image_path: str, model: torch.nn.Module, conf_threshold: float = 0.5, iou_threshold: float = 0.6):
    """Run inference and return annotated image and detections list.

    Returns: (annotated_bgr_image, detections)
    detections: list of (class_name, conf, (x1,y1,x2,y2)) in pixel coordinates
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (C.img_size, C.img_size)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(C.device)

    model.eval()
    outputs = model(tensor)

    def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.6, topk: int = 100) -> List[int]:
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0 and len(keep) < topk:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    H_img, W_img = img_bgr.shape[:2]
    all_boxes: List[Tuple[int, int, int, int]] = []
    all_scores: List[float] = []
    all_cls: List[int] = []

    for pred in outputs:
        N, C_out, H, W = pred.shape
        A = 3
        stride = 5 + C.num_classes
        pred = pred.view(N, A, stride, H, W)
        obj = torch.sigmoid(pred[:, :, 4, :, :])
        tx = torch.sigmoid(pred[:, :, 0, :, :])
        ty = torch.sigmoid(pred[:, :, 1, :, :])
        tw = torch.sigmoid(pred[:, :, 2, :, :])
        th = torch.sigmoid(pred[:, :, 3, :, :])
        cls_logits = pred[:, :, 5:, :, :]
        cls_prob = torch.sigmoid(cls_logits)

        gy, gx = torch.meshgrid(torch.arange(H, device=pred.device), torch.arange(W, device=pred.device), indexing="ij")
        gx = gx[None, None, :, :].float()
        gy = gy[None, None, :, :].float()

        cx = (tx + gx) / max(1, W)
        cy = (ty + gy) / max(1, H)
        ww = tw  # naive width in [0,1]
        hh = th  # naive height in [0,1]

        x1 = (cx - ww / 2.0).clamp(0, 1) * W_img
        y1 = (cy - hh / 2.0).clamp(0, 1) * H_img
        x2 = (cx + ww / 2.0).clamp(0, 1) * W_img
        y2 = (cy + hh / 2.0).clamp(0, 1) * H_img

        cls_prob_flat = cls_prob.permute(0, 1, 3, 4, 2).reshape(-1, C.num_classes)
        obj_flat = obj.reshape(-1)
        boxes_flat = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

        cls_scores, cls_ids = cls_prob_flat.max(dim=1)
        scores = (obj_flat * cls_scores).detach().cpu().numpy()
        boxes_np = boxes_flat.detach().cpu().numpy()
        cls_ids_np = cls_ids.detach().cpu().numpy()

        mask = scores >= conf_threshold
        all_boxes.append(boxes_np[mask])
        all_scores.append(scores[mask])
        all_cls.append(cls_ids_np[mask])

    if all_boxes:
        boxes_cat = np.concatenate(all_boxes, axis=0) if len(all_boxes) > 0 else np.empty((0, 4))
        scores_cat = np.concatenate(all_scores, axis=0) if len(all_scores) > 0 else np.empty((0,))
        cls_cat = np.concatenate(all_cls, axis=0) if len(all_cls) > 0 else np.empty((0,), dtype=int)
    else:
        boxes_cat = np.empty((0, 4))
        scores_cat = np.empty((0,))
        cls_cat = np.empty((0,), dtype=int)

    keep = nms(boxes_cat, scores_cat, iou_thr=iou_threshold, topk=100)
    dets: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for i in keep:
        x1, y1, x2, y2 = boxes_cat[i]
        cls_id = int(cls_cat[i]) if boxes_cat.shape[0] else 0
        name = C.class_names[cls_id] if 0 <= cls_id < len(C.class_names) else str(cls_id)
        conf = float(scores_cat[i])
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(img_bgr, label, (x1i, max(0, y1i - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        dets.append((name, conf, (x1i, y1i, x2i, y2i)))

    return img_bgr, dets


def main():
    parser = argparse.ArgumentParser(description="Minimal inference with demo decoder + NMS")
    parser.add_argument("--source", type=str, default="", help="Image file or directory. If empty, uses dataset test split")
    parser.add_argument("--max", type=int, default=5, help="Max number of images to process (when source is a folder or default dataset")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    args = parser.parse_args()

    weights = os.path.join(C.save_dir, "final.pth")
    model = GCRes2YOLOv3(num_classes=C.num_classes).to(C.device)
    if os.path.exists(weights):
        ckpt = torch.load(weights, map_location=C.device)
        model.load_state_dict(ckpt["model"])  # type: ignore

    # Resolve source list
    paths: List[str] = []
    if args.source:
        if os.path.isdir(args.source):
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for ext in exts:
                paths.extend(sorted(glob.glob(os.path.join(args.source, ext))))
        else:
            paths = [args.source]
    else:
        # Fallback to dataset split
        coco_dir = os.path.join(C.dataset_root, C.test_split)
        yolo_dir = os.path.join(C.dataset_root, "images", C.test_split)
        sample_dir = coco_dir if os.path.exists(os.path.join(coco_dir, "_annotations.coco.json")) else yolo_dir
        paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        paths.sort()

    if args.max and len(paths) > args.max:
        paths = paths[: args.max]

    os.makedirs("runs/detect", exist_ok=True)
    for path in paths:
        name = os.path.basename(path)
        annotated, dets = detect_cow_behavior(path, model, conf_threshold=args.conf, iou_threshold=args.iou)
        out_path = os.path.join("runs/detect", name)
        cv2.imwrite(out_path, annotated)
        print(name, dets)


if __name__ == "__main__":
    main()
