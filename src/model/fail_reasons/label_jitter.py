"""
How steady are the ground-truth boxes themselves?

Why this matters
----------------
The error analysis found that most of the model's "false positives" are
boxes sitting on the real drone with a median IoU of 0.40 against the
label. That is only a model failure if the label is trustworthy.

These labels came from a tracker, not from a person drawing each box.
Trackers drift and breathe: the box grows and shrinks a little every
frame even when the target has not changed size. If the labels wobble
by roughly as much as the model is "wrong" by, then the model is
already at the noise floor and no amount of training fixes the gap.

What is measured
----------------
For each pair of consecutive frames in a clip:

  size change    how much sqrt(w*h) changes, as a fraction
  centre step    how far the box centre moves, in pixels
  IoU            overlap between the two consecutive boxes

A drone's apparent size changes slowly -- it takes many frames to fly
far enough to look meaningfully bigger. So a large frame-to-frame size
change is annotation noise, not motion. The centre can move for real
reasons, which is why it is reported separately rather than mixed in.

The number to compare against is the model's median IoU of 0.40. If
consecutive labels only agree with each other at, say, 0.65, then a
label is not a precise statement about where the drone is, and asking
the model for 0.50 against it is asking for something the data does not
contain.

Usage
-----
    python label_jitter.py
    python label_jitter.py --split validation
    python label_jitter.py --split all
"""

from pathlib import Path
import argparse

import numpy as np

# This script lives one level down from the model code it imports, so
# put the parent directory on the path before importing any of it.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import dataset


def box_iou(a, b):
    """(x, y, w, h) overlap. Local copy so this runs without a model."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter

    return float(inter / union) if union > 0 else 0.0


def clip_jitter(records):
    """
    Walk one clip's frames in order and measure consecutive pairs.

    Only pairs where both frames have exactly one box are used. A frame
    with two boxes would need the boxes matched up first, and a frame
    with none breaks the chain.
    """
    sizes, steps, ious = [], [], []

    prev = None

    for rec in records:
        boxes = rec.get("boxes") or []

        if len(boxes) != 1:
            prev = None
            continue

        cur = [float(v) for v in boxes[0]]

        if prev is not None:
            px, py, pw, ph = prev
            cx, cy, cw, ch = cur

            p_size = np.sqrt(pw * ph)
            c_size = np.sqrt(cw * ch)

            if p_size > 0:
                sizes.append(abs(c_size - p_size) / p_size)

            steps.append(float(np.hypot(
                (cx + cw / 2) - (px + pw / 2),
                (cy + ch / 2) - (py + ph / 2),
            )))

            ious.append(box_iou(prev, cur))

        prev = cur

    return sizes, steps, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation",
                    help="train, validation, test, or all")
    args = ap.parse_args()

    splits = dataset.load_splits()

    if args.split == "all":
        clips = sorted(set(splits["train"] + splits["validation"]
                           + splits["test"]))
    else:
        clips = splits[args.split]

    print()
    print(f"tensors : {config.TENSOR_DIR}")
    print(f"split   : {args.split}  ({len(clips)} clips)")
    print(f"space   : boxes are in {config.INPUT_SIZE}px letterboxed "
          "coordinates")
    print()

    all_sizes, all_steps, all_ious = [], [], []
    rows = []

    for clip in clips:
        _, records = dataset.load_clip(clip)

        if records is None:
            continue

        sizes, steps, ious = clip_jitter(records)

        if not ious:
            continue

        all_sizes += sizes
        all_steps += steps
        all_ious += ious

        rows.append((
            clip,
            len(ious),
            float(np.median(sizes)) if sizes else float("nan"),
            float(np.median(steps)),
            float(np.median(ious)),
            float(np.percentile(ious, 10)),
        ))

    if not rows:
        print("No usable frame pairs. Either the clips are not built or")
        print("no frame has exactly one box.")
        return

    print("PER CLIP")
    print(f"  {'clip':<20} {'pairs':>6} {'size wobble':>12} "
          f"{'centre step':>12} {'IoU med':>9} {'IoU p10':>9}")

    for clip, n, s, st, m, p10 in sorted(rows, key=lambda r: r[4]):
        print(f"  {clip:<20} {n:>6} {s * 100:>11.1f}% "
              f"{st:>11.2f}px {m:>9.3f} {p10:>9.3f}")

    sizes = np.array(all_sizes)
    steps = np.array(all_steps)
    ious = np.array(all_ious)

    print()
    print("=" * 66)
    print("OVERALL")
    print()
    print(f"  consecutive pairs      : {len(ious)}")
    print()
    print(f"  size change  median    : {np.median(sizes) * 100:.1f}%")
    print(f"               p90       : {np.percentile(sizes, 90) * 100:.1f}%")
    print(f"               over 20%  : {(sizes > 0.20).mean() * 100:.1f}% "
          "of pairs")
    print()
    print(f"  centre step  median    : {np.median(steps):.2f} px")
    print(f"               p90       : {np.percentile(steps, 90):.2f} px")
    print()
    print(f"  IoU between consecutive labels")
    print(f"               median    : {np.median(ious):.3f}")
    print(f"               p25       : {np.percentile(ious, 25):.3f}")
    print(f"               p10       : {np.percentile(ious, 10):.3f}")
    print(f"               under 0.5 : {(ious < 0.5).mean() * 100:.1f}% "
          "of pairs")

    print()
    print("=" * 66)
    print("READING THIS")
    print()
    print("  The model's boxes sit on the real drone at a median IoU of")
    print("  0.40. Compare that with the median above.")
    print()
    print("  If consecutive labels agree with each other at well under")
    print("  0.9, the label is not a precise statement about where the")
    print("  drone is, and a 0.5 threshold against it is partly")
    print("  measuring tracker noise rather than model error.")
    print()
    print("  Size wobble is the cleaner signal of the two. A drone")
    print("  cannot change apparent size much between two frames 40ms")
    print("  apart, so a large size change is the annotation moving,")
    print("  not the drone. The centre can move for real reasons.")
    print()
    print("  This does not excuse the model on its own -- both the")
    print("  labels and the model can be loose at the same time. It")
    print("  sets a floor on how good the IoU could get.")


if __name__ == "__main__":
    main()
    