from __future__ import annotations
from typing import List, Tuple
import random
import numpy as np


def bbox_wh_from_labels(labels: List[np.ndarray], img_size: int) -> np.ndarray:
    """Collect (w,h) (in pixels) from YOLO-normalized labels across dataset."""
    wh = []
    for lb in labels:
        if lb.size == 0:
            continue
        wh_pix = lb[:, 2:4] * img_size
        wh.append(wh_pix)
    if not wh:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(wh, axis=0)


def iou_wh(box: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    w, h = box
    iw = np.minimum(w, clusters[:, 0])
    ih = np.minimum(h, clusters[:, 1])
    inter = iw * ih
    union = w * h + clusters[:, 0] * clusters[:, 1] - inter
    return inter / (union + 1e-8)


def kmeans_anchors(wh: np.ndarray, k: int = 9, seed: int = 0) -> np.ndarray:
    """K-means using 1 - IoU as distance for anchor generation."""
    assert wh.ndim == 2 and wh.shape[1] == 2
    if wh.shape[0] < k:
        # fallback to simple grid anchors
        base = np.linspace(10, 100, k)
        return np.stack([base, base], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    centers = wh[rng.choice(wh.shape[0], size=k, replace=False)]
    for _ in range(25):
        sims = np.array([iou_wh(b, centers) for b in wh])  # (N,k)
        clusters = sims.argmax(axis=1)
        new_centers = np.array([wh[clusters == i].mean(axis=0) if np.any(clusters == i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers.astype(np.float32)


def anchors_per_scale(anchors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 9 anchors into 3 scales (sorted by area)."""
    areas = anchors[:, 0] * anchors[:, 1]
    order = np.argsort(areas)
    anchors = anchors[order]
    return anchors[:3], anchors[3:6], anchors[6:9]
