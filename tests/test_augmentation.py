import numpy as np
from project.utils.augmentation import hybrid_augmentation


def test_hybrid_augmentation_shapes():
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)

    class Dummy:
        def rand_index(self):
            return 0
        def load_image_label(self, idx):
            return img.copy(), boxes.copy()

    out_img, out_boxes = hybrid_augmentation(img, boxes, Dummy())
    assert out_img.shape == img.shape
    assert out_boxes.ndim == 2 and out_boxes.shape[1] == 4
