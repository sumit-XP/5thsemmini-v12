from typing import List, Tuple
import random
import numpy as np
import cv2


def horizontal_flip(image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    flipped = cv2.flip(image, 1)
    if boxes.size == 0:
        return flipped, boxes
    boxes = boxes.copy()
    boxes[:, 0] = 1.0 - boxes[:, 0]  # x_center normalized
    return flipped, boxes


def adjust_brightness_hsv(image: np.ndarray, beta: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + beta * 0.2), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def mosaic_augmentation(image: np.ndarray, boxes: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 2x2 mosaic from current and 3 random images from dataset.

    Boxes are remapped; assumes normalized [x,y,w,h].
    """
    img_size = image.shape[0]
    yc, xc = [int(random.uniform(img_size * 0.25, img_size * 0.75)) for _ in range(2)]
    mosaic_img = np.full((img_size * 2, img_size * 2, 3), 114, dtype=np.uint8)
    mosaic_boxes = []

    indices = [dataset.rand_index() for _ in range(3)]
    images = [image]
    labels = [boxes]
    for idx in indices:
        im, lb = dataset.load_image_label(idx)
        images.append(im)
        labels.append(lb)

    positions = ((0, 0, xc, yc), (xc, 0, 2 * img_size, yc), (0, yc, xc, 2 * img_size), (xc, yc, 2 * img_size, 2 * img_size))
    for i, (im, lb) in enumerate(zip(images, labels)):
        h, w = im.shape[:2]
        scale = min((positions[i][2] - positions[i][0]) / w, (positions[i][3] - positions[i][1]) / h)
        im_res = cv2.resize(im, (int(w * scale), int(h * scale)))
        h_res, w_res = im_res.shape[:2]
        x1a, y1a, x2a, y2a = positions[i]
        x2a, y2a = x1a + w_res, y1a + h_res
        mosaic_img[y1a:y2a, x1a:x2a] = im_res

        if lb.size:
            lb = lb.copy()
            # Denormalize to absolute coords
            lb_abs = lb.copy()
            lb_abs[:, 0] *= w
            lb_abs[:, 1] *= h
            lb_abs[:, 2] *= w
            lb_abs[:, 3] *= h
            # Shift to mosaic location
            lb_abs[:, 0] = lb_abs[:, 0] * scale + x1a
            lb_abs[:, 1] = lb_abs[:, 1] * scale + y1a
            lb_abs[:, 2] = lb_abs[:, 2] * scale
            lb_abs[:, 3] = lb_abs[:, 3] * scale
            # Renormalize to mosaic dimensions (2*img_size)
            lb_norm = lb_abs.copy()
            lb_norm[:, 0] /= (2 * img_size)
            lb_norm[:, 1] /= (2 * img_size)
            lb_norm[:, 2] /= (2 * img_size)
            lb_norm[:, 3] /= (2 * img_size)
            mosaic_boxes.append(lb_norm)

    if mosaic_boxes:
        boxes_out = np.concatenate(mosaic_boxes, axis=0)
    else:
        boxes_out = np.zeros((0, 4), dtype=np.float32)

    mosaic_img = cv2.resize(mosaic_img, (img_size, img_size))
    # After resize back to img_size, boxes stay normalized
    return mosaic_img, boxes_out


def hybrid_augmentation(image: np.ndarray, boxes: np.ndarray, dataset, use_mosaic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    rand_val = np.random.randn()
    if rand_val > 0.5:
        image, boxes = horizontal_flip(image, boxes)
    beta = float(np.random.uniform(-1, 1))
    image = adjust_brightness_hsv(image, beta)
    if use_mosaic:
        rand_val = np.random.randn()
        if rand_val > 0.5:
            image, boxes = mosaic_augmentation(image, boxes, dataset)
    return image, boxes
