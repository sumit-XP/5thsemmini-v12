from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_boxes(image_bgr: np.ndarray, boxes_xyxy: List[Tuple[int, int, int, int]], labels: Optional[List[str]] = None, scores: Optional[List[float]] = None) -> np.ndarray:
    out = image_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if labels is not None:
            text = labels[i]
            if scores is not None:
                text = f"{text} {scores[i]:.2f}"
            cv2.putText(out, text, (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=class_names, yticklabels=class_names, title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
