"""Run with: python -m unittest discover tests"""

import unittest

import numpy as np

from defect_detection.config import IMAGE_SIZE, NORMAL_FRACTION
from defect_detection.dataset import generate_dataset, generate_image
from defect_detection.metrics import binary_scores
from defect_detection.rule_based import FEATURE_NAMES, extract_features

N_IMAGES = 40
SEED = 0
CLEAN_SOLIDITY_MIN = 0.97


def features_for(defect, seed=SEED):
    rng = np.random.default_rng(seed)
    values = extract_features(generate_image(rng, defect, "standard"))
    return dict(zip(FEATURE_NAMES, values))


class DatasetTest(unittest.TestCase):
    def test_same_seed_gives_same_images(self):
        first, labels_a, _ = generate_dataset(N_IMAGES, "standard", SEED)
        second, labels_b, _ = generate_dataset(N_IMAGES, "standard", SEED)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(labels_a, labels_b))

    def test_shape_and_class_balance(self):
        images, labels, _ = generate_dataset(N_IMAGES, "harsh", SEED)
        self.assertEqual(images.shape, (N_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual(int((labels == 0).sum()), int(N_IMAGES * NORMAL_FRACTION))


class RuleFeatureTest(unittest.TestCase):
    def test_clean_part_is_convex_and_has_no_dark_pixels(self):
        features = features_for(None)
        self.assertGreater(features["solidity"], CLEAN_SOLIDITY_MIN)
        self.assertEqual(features["dark_ratio"], 0.0)

    def test_notch_lowers_solidity(self):
        self.assertLess(features_for("notch")["solidity"], features_for(None)["solidity"])

    def test_spot_and_scratch_raise_dark_ratio(self):
        self.assertGreater(features_for("spot")["dark_ratio"], 0.0)
        self.assertGreater(features_for("scratch")["dark_ratio"], 0.0)


class MetricsTest(unittest.TestCase):
    def test_precision_recall_f1(self):
        labels = [1, 1, 1, 0, 0]
        predictions = [1, 1, 0, 1, 0]
        scores = binary_scores(labels, predictions)
        self.assertAlmostEqual(scores["precision"], 2 / 3)
        self.assertAlmostEqual(scores["recall"], 2 / 3)
        self.assertAlmostEqual(scores["accuracy"], 3 / 5)


if __name__ == "__main__":
    unittest.main()
