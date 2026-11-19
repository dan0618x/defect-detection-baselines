"""Synthetic images of round parts with or without defects, under controllable lighting.

The data is generated, not photographed. It exists to compare methods under a
known ground truth and a controlled shift in lighting.
"""

import math

import cv2
import numpy as np

from defect_detection.config import (BACKGROUND_LEVEL, CENTER_JITTER, DEFECT_TYPES, IMAGE_SIZE, LIGHTING,
                                     NORMAL_FRACTION, NOTCH_RADIUS, PART_LEVEL, PART_RADIUS_RANGE,
                                     SCRATCH_DEPTH, SCRATCH_HALF_LENGTH, SCRATCH_OFFSET, SCRATCH_THICKNESS,
                                     SPOT_DEPTH, SPOT_OFFSET, SPOT_RADIUS_RANGE)

GRAY_MAX = 255


def random_point_near(rng, center, max_distance):
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(0, max_distance)
    return center[0] + distance * math.cos(angle), center[1] + distance * math.sin(angle)


def draw_defect(canvas, rng, defect, center, radius):
    if defect == "scratch":
        mid = random_point_near(rng, center, SCRATCH_OFFSET * radius)
        angle = rng.uniform(0, math.pi)
        half = SCRATCH_HALF_LENGTH * radius
        start = (round(mid[0] - half * math.cos(angle)), round(mid[1] - half * math.sin(angle)))
        end = (round(mid[0] + half * math.cos(angle)), round(mid[1] + half * math.sin(angle)))
        cv2.line(canvas, start, end, PART_LEVEL - SCRATCH_DEPTH, SCRATCH_THICKNESS)
    elif defect == "notch":
        angle = rng.uniform(0, 2 * math.pi)
        bite = (round(center[0] + radius * math.cos(angle)), round(center[1] + radius * math.sin(angle)))
        cv2.circle(canvas, bite, round(NOTCH_RADIUS * radius), BACKGROUND_LEVEL, -1)
    elif defect == "spot":
        spot = random_point_near(rng, center, SPOT_OFFSET * radius)
        spot_radius = int(rng.integers(SPOT_RADIUS_RANGE[0], SPOT_RADIUS_RANGE[1] + 1))
        cv2.circle(canvas, (round(spot[0]), round(spot[1])), spot_radius, PART_LEVEL - SPOT_DEPTH, -1)


def apply_lighting(canvas, rng, condition):
    """Contrast gain, global brightness offset, a linear gradient and sensor noise."""
    light = LIGHTING[condition]
    gain = rng.uniform(*light["gain"])
    lit = (canvas - BACKGROUND_LEVEL) * gain + BACKGROUND_LEVEL + rng.uniform(-light["offset"], light["offset"])
    angle = rng.uniform(0, 2 * math.pi)
    ys, xs = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    ramp = ((xs - IMAGE_SIZE / 2) * math.cos(angle) + (ys - IMAGE_SIZE / 2) * math.sin(angle)) / IMAGE_SIZE
    lit = lit + ramp * rng.uniform(0, light["gradient"]) + rng.normal(0, light["noise"], lit.shape)
    return np.clip(lit, 0, GRAY_MAX).astype(np.uint8)


def generate_image(rng, defect, condition):
    """Return a BGR uint8 image; defect is None or one of DEFECT_TYPES."""
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_LEVEL, np.float32)
    center = tuple(int(IMAGE_SIZE // 2 + rng.integers(-CENTER_JITTER, CENTER_JITTER + 1)) for _ in range(2))
    radius = int(rng.integers(PART_RADIUS_RANGE[0], PART_RADIUS_RANGE[1] + 1))
    cv2.circle(canvas, center, radius, PART_LEVEL, -1)
    if defect is not None:
        draw_defect(canvas, rng, defect, center, radius)
    return cv2.cvtColor(apply_lighting(canvas, rng, condition), cv2.COLOR_GRAY2BGR)


def defect_plan(n_images, rng):
    """Half normal parts, the rest split evenly over defect types, in random order."""
    n_normal = int(n_images * NORMAL_FRACTION)
    plan = [None] * n_normal
    for index in range(n_images - n_normal):
        plan.append(DEFECT_TYPES[index % len(DEFECT_TYPES)])
    return [plan[index] for index in rng.permutation(n_images)]


def generate_dataset(n_images, condition, seed):
    """Return images (n, H, W, 3) uint8, labels (n,) with 1 = defective, and defect names."""
    rng = np.random.default_rng(seed)
    plan = defect_plan(n_images, rng)
    images = np.stack([generate_image(rng, defect, condition) for defect in plan])
    labels = np.array([0 if defect is None else 1 for defect in plan])
    return images, labels, plan
