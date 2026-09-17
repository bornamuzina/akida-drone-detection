"""
Turn boxes into training targets, and predictions back into boxes.

These two are inverses of each other and share the anchor constants
from config. They are kept in one file for that reason: if the target
builder and the decoder ever disagree about what anchor 2 means, the
network trains correctly and then predicts wrong-sized boxes, which is
a confusing failure to diagnose.

Target layout, matching the model's (7, 7, 30) output reshaped:

    (GRID, GRID, N_ANCHORS, 5 + N_CLASSES)
                             |
                             +-- 0:4  box as (cx, cy, w, h) in cell units
                                 4    objectness (1 if a drone is here)
                                 5:   one-hot class
"""

import numpy as np

import config


# ============================================================
# ANCHOR ASSIGNMENT
# ============================================================

ANCHORS = np.array(config.ANCHORS, dtype=np.float32)      # (5, 2), cell units


def anchor_iou(w, h, anchors=ANCHORS):
    """
    IoU between one box shape and every anchor shape, position ignored.

    Anchors have no location -- they are sizes only -- so both are
    treated as if centred on the same point. This is the standard YOLO
    assignment rule.
    """
    inter_w = np.minimum(w, anchors[:, 0])
    inter_h = np.minimum(h, anchors[:, 1])
    inter = inter_w * inter_h

    union = w * h + anchors[:, 0] * anchors[:, 1] - inter

    return inter / np.maximum(union, 1e-9)


# ============================================================
# BOXES -> TARGET
# ============================================================

def make_target(boxes, grid=None, n_anchors=None, n_classes=None):
    """
    boxes: (n, 4) as (x, y, w, h) in 224-space PIXELS, top-left origin.

    Returns (grid, grid, n_anchors, 5 + n_classes) float32.

    Everything inside is expressed in CELL units, because that is what
    the anchors use and what the network's own outputs are scaled to.
    """
    grid = grid or config.GRID
    n_anchors = n_anchors or config.N_ANCHORS
    n_classes = n_classes or config.N_CLASSES

    target = np.zeros((grid, grid, n_anchors, 5 + n_classes),
                      dtype=np.float32)

    if len(boxes) == 0:
        return target

    cell = config.CELL          # 32 px

    for (x, y, w, h) in boxes:
        # Corner form -> centre form, then pixels -> cell units.
        cx = (x + w / 2) / cell
        cy = (y + h / 2) / cell
        bw = w / cell
        bh = h / cell

        col = int(cx)
        row = int(cy)

        # A centre exactly on the far edge would index out of the grid.
        if not (0 <= col < grid and 0 <= row < grid):
            continue

        # Which anchor shape does this box most resemble?
        ious = anchor_iou(bw, bh)
        best = int(np.argmax(ious))

        # If that slot is already taken by another drone in the same
        # cell, fall back to the next-best free anchor. Without this the
        # second drone would silently overwrite the first -- exactly the
        # crowded-cell case the 7x7 grid is warned about.
        if target[row, col, best, 4] > 0:
            for alt in np.argsort(-ious):
                if target[row, col, int(alt), 4] == 0:
                    best = int(alt)
                    break
            else:
                continue            # cell entirely full; drop this box

        target[row, col, best, 0] = cx
        target[row, col, best, 1] = cy
        target[row, col, best, 2] = bw
        target[row, col, best, 3] = bh
        target[row, col, best, 4] = 1.0
        target[row, col, best, 5] = 1.0         # single class: drone

    return target


def make_targets(boxes_list):
    """Batch version. Returns (batch, grid, grid, anchors, 5 + classes)."""
    return np.stack([make_target(b) for b in boxes_list])


# ============================================================
# TARGET -> BOXES  (the inverse, for checking)
# ============================================================

def decode_target(target):
    """
    Pull boxes back out of a target tensor, in 224-space pixels.

    Used to verify that make_target is lossless. Prediction decoding is
    a different function -- it has to apply sigmoids, exponentials and
    a confidence threshold.
    """
    cell = config.CELL
    boxes = []

    rows, cols, anchors = np.where(target[..., 4] > 0)

    for r, c, a in zip(rows, cols, anchors):
        cx, cy, bw, bh = target[r, c, a, :4]

        w = bw * cell
        h = bh * cell
        x = cx * cell - w / 2
        y = cy * cell - h / 2

        boxes.append((x, y, w, h))

    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


# ============================================================
# CHECK
# ============================================================

def main():
    """
    Verify the round trip on real data.

    A box that goes in and comes out unchanged proves the cell
    arithmetic is right. Boxes that get dropped tell you how often
    crowded cells cost you a label.
    """
    import dataset

    splits = dataset.load_splits()
    ds = dataset.EventDataset(splits["train"], verbose=False)

    if len(ds) == 0:
        print("No samples -- has the tensor generation been run?")
        return

    print(f"samples: {len(ds)}")
    print()

    # ---- round trip -------------------------------------------------
    max_err = 0.0
    n_in = 0
    n_out = 0
    n_multi = 0
    collisions = 0

    rng = np.random.default_rng(0)
    sample = rng.choice(len(ds), min(3000, len(ds)), replace=False)

    for i in sample:
        _, boxes, _ = ds[int(i)]

        if len(boxes) > 1:
            n_multi += 1

        target = make_target(boxes)
        back = decode_target(target)

        n_in += len(boxes)
        n_out += len(back)

        if len(back) < len(boxes):
            collisions += 1

        # Match by nearest centre, since order is not preserved.
        for bx in boxes:
            if len(back) == 0:
                continue
            cx_in = bx[0] + bx[2] / 2
            cy_in = bx[1] + bx[3] / 2
            cx_out = back[:, 0] + back[:, 2] / 2
            cy_out = back[:, 1] + back[:, 3] / 2

            d = np.hypot(cx_out - cx_in, cy_out - cy_in)
            j = int(np.argmin(d))

            max_err = max(max_err, float(np.abs(back[j] - bx).max()))

    print(f"round trip over {len(sample)} samples")
    print(f"  boxes in     : {n_in}")
    print(f"  boxes out    : {n_out}")
    print(f"  max error    : {max_err:.6f} px")
    print(f"  frames with >1 box : {n_multi}")
    print(f"  frames losing a box: {collisions}")
    print()

    if max_err < 1e-3:
        print("  Encoding is lossless.")
    else:
        print("  Boxes are changing. The cell arithmetic is wrong.")

    if collisions:
        print(f"  {collisions} frames had two drones too close to separate")
        print("  even across five anchors -- a real cost of the 7x7 grid.")

    # ---- anchor usage -----------------------------------------------
    print()
    print("anchor assignment across the sample:")

    counts = np.zeros(config.N_ANCHORS, dtype=int)

    for i in sample:
        _, boxes, _ = ds[int(i)]
        for (x, y, w, h) in boxes:
            ious = anchor_iou(w / config.CELL, h / config.CELL)
            counts[int(np.argmax(ious))] += 1

    for a in range(config.N_ANCHORS):
        px = np.array(config.ANCHORS[a]) * config.CELL
        share = counts[a] / max(counts.sum(), 1) * 100
        print(f"  anchor {a} ({px[0]:5.1f} x {px[1]:5.1f} px): "
              f"{counts[a]:6d}  {share:5.1f}%")

    unused = [a for a in range(config.N_ANCHORS) if counts[a] == 0]
    if unused:
        print(f"  anchors never used: {unused}")

    # ---- occupancy ---------------------------------------------------
    # How much of the 245-slot output is ever positive? Extreme imbalance
    # between positive and negative slots is the main difficulty in
    # training a detector, and it is worth knowing the number.
    target = make_targets([ds[int(i)][1] for i in sample[:500]])
    positives = target[..., 4].sum()
    total = target[..., 4].size

    print()
    print(f"positive slots: {positives:.0f} of {total} "
          f"({positives / total * 100:.3f}%)")
    print("  The loss has to cope with that imbalance -- almost every")
    print("  slot should predict 'nothing here'.")


if __name__ == "__main__":
    main()