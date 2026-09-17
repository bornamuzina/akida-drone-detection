"""
Choose the clip point for mapping signed accumulations to uint8.

The model's Rescaling layer expects 0-255 input, and Akida's input layer
is 8-bit, so the tensors have to become integers. Mapping the full range
would waste almost all of the 256 levels on rare extreme pixels.

This measures where the values actually live, then reports what each
candidate clip point would cost in saturation and what resolution it
would buy.

Usage:
    python measure_clip.py
"""

import numpy as np

import config
import dataset


CANDIDATES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def main():
    splits = dataset.load_splits()
    ds = dataset.EventDataset(splits["train"], verbose=False)

    if len(ds) == 0:
        print("No samples found.")
        return

    rng = np.random.default_rng(0)
    picks = rng.choice(len(ds), min(400, len(ds)), replace=False)

    print(f"sampling {len(picks)} frames from {len(ds)}")

    # Non-zero values only. Zeros dominate (~87% of pixels) and would
    # swamp every percentile, but they carry no information about where
    # to clip -- they map to the midpoint regardless.
    vals = []
    n_pixels = 0
    n_zero = 0

    for i in picks:
        img, _, _ = ds[int(i)]
        flat = img.ravel()
        n_pixels += flat.size
        z = flat == 0
        n_zero += int(z.sum())
        vals.append(flat[~z])

    vals = np.concatenate(vals)

    print(f"pixels        : {n_pixels:,}")
    print(f"zero          : {n_zero / n_pixels * 100:.1f}%")
    print(f"non-zero      : {len(vals):,}")
    print()

    a = np.abs(vals)

    print("magnitude of non-zero pixels:")
    for q in (50, 75, 90, 95, 99, 99.9, 100):
        print(f"  p{q:<5} {np.percentile(a, q):8.3f}")
    print()

    print("range:", f"{vals.min():.2f} to {vals.max():.2f}")
    print()

    # ---- what each clip point would do -------------------------------
    print("clip   saturated   levels used by p50   levels by p99")
    print("-" * 55)

    for c in CANDIDATES:
        sat = (a > c).mean() * 100

        # How many of the 256 levels a typical pixel is separated by.
        # Too few and distinct event counts become the same integer.
        step = (2 * c) / 255
        lv50 = np.percentile(a, 50) / step
        lv99 = np.percentile(a, 99) / step

        print(f"{c:5.1f}   {sat:7.3f}%   {lv50:16.1f}   {lv99:13.1f}")

    print()
    print("Saturation is the cost: those pixels lose their true value.")
    print("Levels is the benefit: how finely typical pixels are separated.")
    print()

    # ---- how much of the drone signal would saturate? -----------------
    # Pixels inside the boxes matter more than background ones. If a clip
    # point saturates the drone's own edges it is doing real damage.
    inside = []

    for i in picks[:150]:
        img, boxes, _ = ds[int(i)]
        img = img[..., 0]

        for (x, y, w, h) in boxes:
            x0, y0 = int(max(0, x)), int(max(0, y))
            x1, y1 = int(min(224, x + w)), int(min(224, y + h))
            if x1 <= x0 or y1 <= y0:
                continue
            patch = img[y0:y1, x0:x1].ravel()
            inside.append(patch[patch != 0])

    if inside:
        inside = np.abs(np.concatenate(inside))
        print(f"non-zero pixels inside boxes: {len(inside):,}")
        print(f"  median {np.median(inside):.3f}  "
              f"p95 {np.percentile(inside, 95):.3f}  "
              f"max {inside.max():.3f}")
        print()
        print("clip   drone pixels saturated")
        for c in CANDIDATES:
            print(f"{c:5.1f}   {(inside > c).mean() * 100:7.3f}%")

    print()
    print("Pick the smallest clip that leaves drone-pixel saturation")
    print("negligible. Smaller clip = more resolution where the data is.")


if __name__ == "__main__":
    main()