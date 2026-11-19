"""All tunable numbers in one place."""

SEED = 0

# Synthetic image geometry (grayscale intensities 0-255)
IMAGE_SIZE = 96
BACKGROUND_LEVEL = 70
PART_LEVEL = 190
PART_RADIUS_RANGE = (26, 32)
CENTER_JITTER = 6

# Defects
DEFECT_TYPES = ["scratch", "notch", "spot"]
NORMAL_FRACTION = 0.5
SCRATCH_DEPTH = 80            # how much darker a scratch is than the part
SCRATCH_THICKNESS = 2
SCRATCH_OFFSET = 0.4          # scratch centre within 0.4 r of the part centre
SCRATCH_HALF_LENGTH = 0.5     # half length as a fraction of r
NOTCH_RADIUS = 0.3            # bite radius as a fraction of r
SPOT_DEPTH = 90
SPOT_RADIUS_RANGE = (3, 5)
SPOT_OFFSET = 0.6

# Lighting conditions: contrast gain range, brightness offset, gradient amplitude, noise std
LIGHTING = {
    "standard": {"gain": (0.9, 1.0), "offset": 10, "gradient": 15, "noise": 4},
    "harsh": {"gain": (0.35, 1.0), "offset": 40, "gradient": 90, "noise": 14},
}

# Dataset sizes
N_TRAIN = 1600
N_TEST = 400

# Rule-based detector
BLUR_KERNEL = (5, 5)
ERODE_KERNEL = (5, 5)
ERODE_ITERATIONS = 2
DARK_DELTA = 40               # pixel counts as "dark" if this much below the part median
SOLIDITY_GRID = (0.90, 0.995, 20)
DARK_RATIO_GRID = (0.0, 0.05, 51)
CANNY_LOW = 50
CANNY_HIGH = 150

# Random forest
RF_TREES = 200

# CNN
CNN_EPOCHS = 20
CNN_BATCH = 32
CNN_LEARNING_RATE = 1e-3
CNN_FILTERS = [16, 32, 64]
DECISION_THRESHOLD = 0.5
