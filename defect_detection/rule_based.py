"""Classic OpenCV pipeline: segment the part, measure its shape and surface, apply thresholds.

- notch   -> the outline is no longer convex  -> low solidity
- scratch -> dark pixels inside the part      -> high dark ratio
- spot    -> dark pixels inside the part      -> high dark ratio
"""

import math

import cv2
import numpy as np

from defect_detection.config import (BLUR_KERNEL, CANNY_HIGH, CANNY_LOW, DARK_DELTA, DARK_RATIO_GRID,
                                     ERODE_ITERATIONS, ERODE_KERNEL, SOLIDITY_GRID)
from defect_detection.metrics import binary_scores

GRAY_MAX = 255
FEATURE_NAMES = ["solidity", "circularity", "dark_ratio", "area_ratio", "inner_std", "edge_density"]
# Returned when no part is found: looks maximally defective
MISSING_PART_FEATURES = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def segment_part(gray):
    """Otsu threshold, then keep the largest outer contour as the part."""
    blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    _, mask = cv2.threshold(blur, 0, GRAY_MAX, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return blur, None
    return blur, max(contours, key=cv2.contourArea)


def shape_features(contour, image_area):
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    perimeter = cv2.arcLength(contour, True)
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = 4 * math.pi * area / perimeter ** 2 if perimeter > 0 else 0.0
    return solidity, circularity, area / image_area


def surface_features(gray, blur, contour):
    """Look only inside the part (eroded so the bright/dark border is excluded)."""
    mask = np.zeros_like(blur)
    cv2.drawContours(mask, [contour], -1, GRAY_MAX, cv2.FILLED)
    inner = cv2.erode(mask, np.ones(ERODE_KERNEL, np.uint8), iterations=ERODE_ITERATIONS) > 0
    if inner.sum() == 0:
        return 1.0, 0.0, 1.0
    pixels = blur[inner].astype(np.float32)
    dark_ratio = float(np.mean(pixels < np.median(pixels) - DARK_DELTA))
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0
    return dark_ratio, float(pixels.std()), float(edges[inner].mean())


def extract_features(image):
    """BGR image -> list of FEATURE_NAMES values."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur, contour = segment_part(gray)
    if contour is None:
        return list(MISSING_PART_FEATURES)
    solidity, circularity, area_ratio = shape_features(contour, gray.size)
    dark_ratio, inner_std, edge_density = surface_features(gray, blur, contour)
    return [solidity, circularity, dark_ratio, area_ratio, inner_std, edge_density]


def features_of(images):
    return np.array([extract_features(image) for image in images])


def predict_rule(features, solidity_min, dark_ratio_max):
    solidity = features[:, FEATURE_NAMES.index("solidity")]
    dark_ratio = features[:, FEATURE_NAMES.index("dark_ratio")]
    return ((solidity < solidity_min) | (dark_ratio > dark_ratio_max)).astype(int)


def tune_rule(features, labels):
    """Grid-search the two thresholds for the best F1 on the training set."""
    best = (0.0, None, None)
    for solidity_min in np.linspace(*SOLIDITY_GRID):
        for dark_ratio_max in np.linspace(*DARK_RATIO_GRID):
            f1 = binary_scores(labels, predict_rule(features, solidity_min, dark_ratio_max))["f1"]
            if f1 > best[0]:
                best = (f1, float(solidity_min), float(dark_ratio_max))
    return best[1], best[2]
