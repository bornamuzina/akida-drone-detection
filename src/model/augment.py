"""
Augmentation that moves the boxes with the pixels.

The one rule
------------
Every transform here returns BOTH the image and the boxes. A transform
that changes pixels without changing coordinates does not raise an
error -- it quietly teaches the model that the drone is somewhere it is
not, and the run simply comes out worse for no visible reason. That is
the failure this module is written to avoid, and why every function
takes and returns a pair.

Boxes are (x, y, w, h) in pixels, top-left origin, same space as
dataset.EventDataset yields. A frame with no drone has shape (0, 4) and
passes through every transform unchanged, which is what negatives need.

What is here, and what is deliberately not
------------------------------------------
horizontal flip   safe. A drone mirrored is a drone.

brightness        safe, and the clips span very different light.

contrast          same.

translate         small shifts. Also gives the box head more offsets to
                  see, which is where the model is weakest.

scale             shrink the frame and pad it back. A 14 px drone
                  becomes an 8 px drone. This manufactures the small
                  targets the dataset is short of -- recall under 8 px
                  is 0.106 -- so of everything here it is the one aimed
                  at a measured weakness rather than at variety in
                  general.

NO vertical flip. The band analysis showed position in frame matters a
great deal: against sky the model finds drones and fumbles the box,
against terrain it misses them. Sky above and ground below is real
structure, and flipping it teaches something untrue.

NO rotation beyond a degree or two, for the same reason, and because
rotating an axis-aligned box either grows it or clips the target.

Usage
-----
    from augment import augment

    image, boxes = augment(image, boxes, rng)

Called only when training. Validation and test must stay untouched or
the numbers stop meaning anything.
"""

import cv2
import numpy as np

import config


# Defaults. Deliberately mild: augmentation that distorts more than the
# real variation in the data teaches the model to be robust to things
# that never happen, at the cost of accuracy on things that do.
P_FLIP = 0.5
P_BRIGHTNESS = 0.5
P_CONTRAST = 0.3
P_TRANSLATE = 0.3
P_SCALE = 0.3

BRIGHTNESS = 40.0        # +/- this many 0-255 levels
CONTRAST = (0.8, 1.25)   # multiplier
TRANSLATE = 0.06         # fraction of the frame
SCALE = (0.6, 1.0)       # shrink only -- growing a drone is not the gap


def _clip_boxes(boxes, size):
    """
    Keep boxes inside the frame, and drop what has left it.

    A box pushed mostly out of frame by a shift is worse than no box:
    the drone is barely visible but the model is told to find it there.
    So anything that loses over half its area goes.
    """
    if len(boxes) == 0:
        return boxes

    out = []

    for (x, y, w, h) in boxes:
        area = max(w * h, 1e-6)

        x0 = max(0.0, x)
        y0 = max(0.0, y)
        x1 = min(float(size), x + w)
        y1 = min(float(size), y + h)

        if x1 <= x0 or y1 <= y0:
            continue

        if (x1 - x0) * (y1 - y0) < 0.5 * area:
            continue

        out.append((x0, y0, x1 - x0, y1 - y0))

    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


# ============================================================
# TRANSFORMS
# ============================================================

def flip_horizontal(image, boxes):
    size = image.shape[1]

    image = image[:, ::-1]

    if len(boxes):
        boxes = boxes.copy()
        # The mirrored box starts where the old one ended.
        boxes[:, 0] = size - (boxes[:, 0] + boxes[:, 2])

    return image, boxes


def brightness(image, boxes, rng, amount=BRIGHTNESS):
    delta = rng.uniform(-amount, amount)
    return np.clip(image + delta, 0, 255), boxes


def contrast(image, boxes, rng, span=CONTRAST):
    factor = rng.uniform(*span)
    # Around the mid-grey point, so the image darkens and lightens
    # symmetrically rather than drifting.
    return np.clip((image - 128.0) * factor + 128.0, 0, 255), boxes


def translate(image, boxes, rng, amount=TRANSLATE):
    size = image.shape[0]
    max_shift = amount * size

    dx = rng.uniform(-max_shift, max_shift)
    dy = rng.uniform(-max_shift, max_shift)

    m = np.float32([[1, 0, dx], [0, 1, dy]])

    image = cv2.warpAffine(image, m, (size, size),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if image.ndim == 2:
        image = image[..., None]

    if len(boxes):
        boxes = boxes.copy()
        boxes[:, 0] += dx
        boxes[:, 1] += dy
        boxes = _clip_boxes(boxes, size)

    return image, boxes


def scale(image, boxes, rng, span=SCALE):
    """
    Shrink the whole frame and pad it back to size.

    This is the one aimed at a known gap rather than at variety: a
    drone at 14 px becomes one at 8 px, and sub-8px recall is where the
    model fails hardest. INTER_AREA on the way down, for the same
    reason the build pipeline uses it -- it averages the source pixels
    instead of sampling one of them.
    """
    size = image.shape[0]
    f = rng.uniform(*span)

    new = max(int(round(size * f)), 16)

    small = cv2.resize(image, (new, new), interpolation=cv2.INTER_AREA)

    if small.ndim == 2:
        small = small[..., None]

    out = np.zeros_like(image)

    # Place it somewhere at random rather than always centred, or the
    # model learns that small drones live in the middle.
    x0 = int(rng.integers(0, size - new + 1))
    y0 = int(rng.integers(0, size - new + 1))

    out[y0:y0 + new, x0:x0 + new] = small

    if len(boxes):
        boxes = boxes.copy() * f
        boxes[:, 0] += x0
        boxes[:, 1] += y0
        boxes = _clip_boxes(boxes, size)

    return out, boxes


# ============================================================
# ENTRY POINT
# ============================================================

def augment(image, boxes, rng,
            p_flip=P_FLIP, p_brightness=P_BRIGHTNESS, p_contrast=P_CONTRAST,
            p_translate=P_TRANSLATE, p_scale=P_SCALE):
    """
    Apply a random subset. Returns (image, boxes), both float32.

    Geometry first, then photometry: a shift has to happen before the
    pixel values are adjusted, or the padding introduced by the shift
    gets brightened along with the image and stops being black.
    """
    image = np.asarray(image, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    if rng.random() < p_flip:
        image, boxes = flip_horizontal(image, boxes)

    if rng.random() < p_scale:
        image, boxes = scale(image, boxes, rng)

    if rng.random() < p_translate:
        image, boxes = translate(image, boxes, rng)

    if rng.random() < p_brightness:
        image, boxes = brightness(image, boxes, rng)

    if rng.random() < p_contrast:
        image, boxes = contrast(image, boxes, rng)

    return np.ascontiguousarray(image, dtype=np.float32), boxes


# ============================================================
# CHECK
# ============================================================

def main():
    """
    Draw the boxes on augmented frames and write them out.

    The whole risk in this file is boxes drifting away from the pixels,
    and no assertion catches that as well as looking. Green boxes should
    sit on the drone in every tile.
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="augment_check.jpg")
    ap.add_argument("--zoom", type=int, default=1)
    args = ap.parse_args()

    import dataset

    splits = dataset.load_splits()
    ds = dataset.EventDataset(splits["train"], verbose=False)

    if len(ds) == 0:
        print("No samples.")
        return

    rng = np.random.default_rng(0)

    cols = 6
    rows = (args.n + cols - 1) // cols
    size = config.INPUT_SIZE

    sheet = np.full((rows * size, cols * size, 3), 30, dtype=np.uint8)

    n_before = 0
    n_after = 0

    for i in range(args.n):
        idx = int(rng.integers(0, len(ds)))
        image, boxes, _ = ds[idx]

        n_before += len(boxes)

        image, boxes = augment(image, boxes, rng)

        n_after += len(boxes)

        tile = np.clip(image, 0, 255).astype(np.uint8)
        if tile.ndim == 2 or tile.shape[2] == 1:
            tile = cv2.cvtColor(tile.reshape(size, size), cv2.COLOR_GRAY2BGR)

        for (x, y, w, h) in boxes:
            cv2.rectangle(tile,
                          (int(round(x)), int(round(y))),
                          (int(round(x + w)), int(round(y + h))),
                          (0, 255, 0), 1)

        r, c = divmod(i, cols)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = tile

    if args.zoom > 1:
        sheet = cv2.resize(sheet, None, fx=args.zoom, fy=args.zoom,
                           interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"wrote {args.out}")
    print(f"boxes in  : {n_before}")
    print(f"boxes out : {n_after}"
          f"  ({n_before - n_after} left the frame and were dropped)")
    print()
    print("Look at it. Every green box should sit on a drone. If any")
    print("float in empty sky, a transform is moving pixels without")
    print("moving coordinates, and training on that would be worse than")
    print("not augmenting at all.")


if __name__ == "__main__":
    main()