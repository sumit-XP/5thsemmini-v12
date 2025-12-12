import numpy as np
from project.utils.anchors import kmeans_anchors, anchors_per_scale


def test_anchor_generation_shapes():
    wh = np.array([[20, 30], [40, 50], [10, 12], [33, 60], [100, 80], [22, 18], [12, 15], [60, 70], [16, 12], [55, 33]], dtype=np.float32)
    anchors = kmeans_anchors(wh, k=9)
    assert anchors.shape == (9, 2)
    a1, a2, a3 = anchors_per_scale(anchors)
    assert a1.shape == (3, 2) and a2.shape == (3, 2) and a3.shape == (3, 2)
