# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmseg.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import points2boundary, box_score_fast, unclip

@POSTPROCESSOR.register_module()
class DBPostprocessor(BasePostprocessor):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        max_candidates (int): The maximum candidate number.
    """

    def __init__(self,
                 text_repr_type='poly',
                 mask_thr=0.5,
                 min_text_score=0.5,
                 min_text_width=3,
                 unclip_ratio=0.,
                 max_candidates=300,
                 **kwargs):
        super().__init__(text_repr_type)
        self.mask_thr = mask_thr
        self.min_text_score = min_text_score
        self.min_text_width = min_text_width
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(N, C=3, H, W)`.

        Returns:
            list[list[float]]: The predicted text boundaries.
        """
        N, C, H, W = preds.size()
        assert(C == 3)
        preds = preds.data.cpu().numpy().astype(np.float32)
        preds = np.exp(preds)

        masks = []
        for pred in preds:
            fg_bg = pred.copy()
            fg_bg = np.divide(fg_bg, np.repeat(np.sum(fg_bg, axis=0, keepdims=True), 3, axis=0))
            fg_bg[1, ...] += fg_bg[2, ...]
            fg_boundaries = self.parse_each_map(fg_bg[[1,0], :, :].squeeze())

            # boundaries to map
            mask = np.zeros((H, W), dtype=np.uint8)
            mask = cv2.polylines(mask, fg_boundaries, 1, 1)
            mask = cv2.fillPoly(mask, fg_boundaries, 1)
            
            mgs_bg = pred.copy()[1:,...]
            mgs_bg = np.divide(mgs_bg, np.repeat(np.sum(mgs_bg, axis=0, keepdims=True), 2, axis=0))
            #mgs_bg[0, ...] = np.divide(mgs_bg[1, ...], fg_bg[1, ...])
            #mgs_bg[1, ...] = np.divide(mgs_bg[2, ...], fg_bg[1, ...])
            mgs_bg[0][mask == 0] = 1
            mgs_bg[1][mask == 0] = 0.
            mgs_coal_boundaries = self.parse_each_map(mgs_bg[[1,0],:,:])

            # boundaries to map
            mask = cv2.polylines(mask, mgs_coal_boundaries, 1, 2)
            mask = cv2.fillPoly(mask, mgs_coal_boundaries, 2)
            
            masks.append(mask)

        return np.array(masks)

    def parse_each_map(self, preds, ratio=0.001, mask_thr=0.5):
        """
        Args:
            preds numpy array: Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The predicted text boundaries.
        """
        score_map = preds[0, :, :]
        text_mask = (score_map > mask_thr).astype(np.uint8)

        #score_map = prob_map.data.cpu().numpy().astype(np.float32)
        #text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

        contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boundaries = []
        for i, poly in enumerate(contours):
            if i > self.max_candidates:
                break
            epsilon = ratio * cv2.arcLength(poly, True)
            approx = cv2.approxPolyDP(poly, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = box_score_fast(score_map, points)
            if score < self.min_text_score:
                continue
            poly = unclip(points, unclip_ratio=self.unclip_ratio)
            if len(poly) == 0 or isinstance(poly[0], list):
                continue
            poly = poly.reshape(-1, 2)

            if self.text_repr_type == 'quad':
                poly = points2boundary(poly, self.text_repr_type, score,
                                       self.min_text_width)
                poly = np.array(poly).reshape(-1,2)
            elif self.text_repr_type == 'poly':
                if len(poly) < 3:
                    poly = None

            if poly is not None:
                boundaries.append(poly)

        return boundaries
