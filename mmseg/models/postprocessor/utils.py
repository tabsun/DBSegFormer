# Copyright (c) OpenMMLab. All rights reserved.
import functools
import operator

import cv2
import numpy as np
import pyclipper
from numpy.fft import ifft
from numpy.linalg import norm
from shapely.geometry import Polygon

def points2boundary(points, text_repr_type, text_score=None, min_width=-1):
    """Convert a text mask represented by point coordinates sequence into a
    text boundary.
    Args:
        points (ndarray): Mask index of size (n, 2).
        text_repr_type (str): Text instance encoding type
            ('quad' for quadrangle or 'poly' for polygon).
        text_score (float): Text score.
    Returns:
        boundary (list[float]): The text boundary point coordinates (x, y)
            list. Return None if no text boundary found.
    """
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 2
    assert text_repr_type in ['quad', 'poly']
    assert text_score is None or 0 <= text_score <= 1

    if text_repr_type == 'quad':
        rect = cv2.minAreaRect(points)
        vertices = cv2.boxPoints(rect)
        boundary = []
        if min(rect[1]) > min_width:
            boundary = [p for p in vertices.flatten().tolist()]

    elif text_repr_type == 'poly':

        height = np.max(points[:, 1]) + 10
        width = np.max(points[:, 0]) + 10

        mask = np.zeros((height, width), np.uint8)
        mask[points[:, 1], points[:, 0]] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boundary = list(contours[0].flatten().tolist())

    if text_score is not None:
        boundary = boundary + [text_score]
    if len(boundary) < 8:
        return None

    return boundary

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded
