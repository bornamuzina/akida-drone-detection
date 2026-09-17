"""
Every constant that has to agree across the pipeline.

Anchors in particular must be identical in target creation and in
inference decoding. If they drift apart the network trains fine and
then predicts boxes of the wrong size, which looks like a broken model
rather than a broken constant. Keeping them in one file that both sides
import removes that failure mode.
"""

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# Resolve from this file's location: src/model/config.py -> project root.
# An env var overrides it, for anyone whose data lives elsewhere.
import os
ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))

# TENSOR_DIR = ROOT / "Data_new" / "tensors_rgb" / "tensors_rgb_224_packed"
# TENSOR_DIR = ROOT / "Data_new" / "tensors_events" / "tensors_messy_20ms_packed"
# TENSOR_DIR = ROOT / "Data_new" / "tensors_rgb" / "tensors_rgb_448_packed"
TENSOR_DIR = ROOT / "Data_new" / "tensors_rgb" / "tensors_rgb_256_packed"
SPLITS_FILE = ROOT / "Data_new" / "splits.json"
RUNS_DIR = ROOT / "Data_new" / "runs"


# ============================================================
# INPUT GEOMETRY
# ============================================================

#INPUT_SIZE = 224
INPUT_SIZE = 256
#INPUT_SIZE = 448
CHANNELS = 3              # RGB; for event tensors use 1
# CHANNELS = 1
# GRID = 7
#GRID = 14
GRID = 16
CELL = INPUT_SIZE / GRID  # 32 px either way

SRC_W, SRC_H = 640, 512

# Letterbox: same scale on both axes, leftover padded with zeros.
SCALE = min(INPUT_SIZE / SRC_W, INPUT_SIZE / SRC_H)     # 0.70 at 448
PAD_X = (INPUT_SIZE - SRC_W * SCALE) / 2                # 0.0
PAD_Y = (INPUT_SIZE - SRC_H * SCALE) / 2                # 44.8


# ============================================================
# ANCHORS
# ============================================================
#
# Measured by IoU k-means over all 36,316 boxes in the 114 label CSVs.
# Mean IoU 0.818 at both resolutions.
#
# Units are GRID CELLS, which is what akida_models expects -- not pixels.
# Multiply by CELL (32) to get pixels. The cell is 32 px at either
# resolution, so doubling the input doubles the anchors in cell units.

# All four sets come from the same 36,316 boxes, mean IoU 0.818 every
# time. Units are GRID CELLS, which is what akida_models expects, so
# multiply by CELL to get pixels. The cell is 32 px at stride 32 and
# 16 px at stride 16, which is why 224 appears twice with different
# numbers.

# 224 input, stock stride 32, 7x7 grid, cell 32 px. Median 0.33 cells.
# ANCHORS = [
#     [0.2646, 0.2147],   # 8.47 x 6.87 px    26.7% of boxes
#     [0.3548, 0.2815],   # 11.35 x 9.01 px   36.3%
#     [0.4629, 0.3301],   # 14.81 x 10.56 px  23.8%
#     [0.6489, 0.4459],   # 20.76 x 14.27 px  11.7%
#     [0.9655, 0.7012],   # 30.90 x 22.44 px   1.5%
# ]

# 448 input, stock stride 32, 14x14 grid, cell 32 px. Median 0.65 cells.
# Not deployable: AKD1500 caps input at 256.
# ANCHORS = [
#     [0.5293, 0.4294],   # 16.94 x 13.74 px  26.7% of boxes
#     [0.7096, 0.5630],   # 22.71 x 18.02 px  36.3%
#     [0.9259, 0.6601],   # 29.63 x 21.12 px  23.8%
#     [1.2978, 0.8918],   # 41.53 x 28.54 px  11.7%
#     [1.9311, 1.4024],   # 61.80 x 44.88 px   1.5%
# ]

# 224 input, stride 16, 14x14 grid, cell 16 px. Median 0.65 cells.
# Same cell-unit values as the 448 set: half the pixels, half the cell.
# ANCHORS = [
#     [0.5293, 0.4294],   # 8.47 x 6.87 px    26.7% of boxes
#     [0.7096, 0.5630],   # 11.35 x 9.01 px   36.3%
#     [0.9259, 0.6601],   # 14.81 x 10.56 px  23.8%
#     [1.2978, 0.8918],   # 20.76 x 14.27 px  11.7%
#     [1.9311, 1.4024],   # 30.90 x 22.44 px   1.5%
# ]

# 256 input, stride 16, 16x16 grid, cell 16 px. Median 0.74 cells.
# Largest input AKD1500 accepts. Sub-8px boxes drop to 6.6% from 15.9%.
ANCHORS = [
    [0.6049, 0.4908],     # 9.68 x 7.85 px    26.7% of boxes
    [0.8110, 0.6435],     # 12.98 x 10.30 px  36.3%
    [1.0581, 0.7544],     # 16.93 x 12.07 px  23.8%
    [1.4832, 1.0192],     # 23.73 x 16.31 px  11.7%
    [2.2070, 1.6028],     # 35.31 x 25.64 px   1.5%
]

N_ANCHORS = len(ANCHORS)

# ============================================================
# CLASSES
# ============================================================

CLASSES = ["drone"]
N_CLASSES = len(CLASSES)

# Output depth: per anchor, 4 box numbers + objectness + class scores.
OUTPUT_DEPTH = N_ANCHORS * (4 + 1 + N_CLASSES)          # 30


# ============================================================
# REPRESENTATION
# ============================================================

WINDOW_MS = 20.0          # the guide's practical baseline
PRESET = "messy"          # Cp/Cn 0.05, shot noise 15 Hz

# Frames whose boxes contain fewer than this many events. A correct box
# over a near-stationary drone can legitimately contain nothing.
QUIET_THRESHOLD = 10


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3

# What to do with quiet frames:
#   "keep"   train on them as normal positives
#   "drop"   exclude them entirely
#   "ignore" keep the image, exclude the box from the loss
QUIET_POLICY = "keep"