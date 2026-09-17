"""
Work out what shape the drones actually are, and pick five anchors.

Anchors are prior box shapes. They should describe THIS dataset, not
VOC's cars and people. Everything here is measured from the exported
label CSVs.

All sizes are reported in the model's 224x224 letterboxed space, since
that is where the anchors are used -- not in original video pixels.

Usage:
    python compute_anchors.py --labels C:\\Users\\borna\\Desktop\\Dronovi\\Data_new\\labels
"""

from pathlib import Path
import argparse
import csv

import numpy as np


# The letterbox transform. Source is 640x512, model input is 224x224.
# Scale is the same on both axes so shapes are preserved; the leftover
# rows become zero padding.
SRC_W, SRC_H = 640, 512
DST = 256
GRID = 16


def letterbox_params(src_w, src_h, dst):
    """
    Returns (scale, pad_x, pad_y) for fitting src into a dst x dst square
    without distorting it.
    """
    scale = min(dst / src_w, dst / src_h)
    new_w, new_h = src_w * scale, src_h * scale
    return scale, (dst - new_w) / 2, (dst - new_h) / 2


def load_all_boxes(labels_dir):
    """
    Read every CSV and return box widths/heights in ORIGINAL pixels,
    plus a per-clip count so we can report coverage.
    """
    wh = []
    per_clip = {}

    for path in sorted(Path(labels_dir).glob("*.csv")):
        n = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                w = float(row["w"])
                h = float(row["h"])
                if w > 0 and h > 0:
                    wh.append((w, h))
                    n += 1
        per_clip[path.stem] = n

    return np.array(wh, dtype=np.float64), per_clip


def iou_wh(boxes, clusters):
    """
    IoU between box shapes and cluster shapes, ignoring position.
    Both are (w, h) only -- anchors have no location.
    """
    w = np.minimum(boxes[:, None, 0], clusters[None, :, 0])
    h = np.minimum(boxes[:, None, 1], clusters[None, :, 1])

    inter = w * h
    area_b = (boxes[:, 0] * boxes[:, 1])[:, None]
    area_c = (clusters[:, 0] * clusters[:, 1])[None, :]

    return inter / (area_b + area_c - inter)


def kmeans_iou(boxes, k, seed=0, max_iter=200):
    """
    k-means clustering using 1 - IoU as the distance.

    Plain Euclidean k-means over (w, h) biases towards large boxes.
    IoU distance is what the YOLO papers use and it is scale-aware.
    """
    rng = np.random.default_rng(seed)

    # Start from k randomly chosen real boxes.
    idx = rng.choice(len(boxes), k, replace=False)
    clusters = boxes[idx].copy()

    last = None

    for _ in range(max_iter):
        d = 1.0 - iou_wh(boxes, clusters)
        assign = d.argmin(axis=1)

        if last is not None and (assign == last).all():
            break
        last = assign

        for j in range(k):
            members = boxes[assign == j]
            if len(members):
                clusters[j] = members.mean(axis=0)

    mean_iou = float(iou_wh(boxes, clusters)[np.arange(len(boxes)), assign].mean())

    return clusters, assign, mean_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="folder of frame,x,y,w,h CSVs")
    ap.add_argument("--k", type=int, default=5, help="number of anchors")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    boxes_src, per_clip = load_all_boxes(args.labels)

    if len(boxes_src) == 0:
        raise SystemExit("No boxes found. Check the --labels path.")

    empty = [c for c, n in per_clip.items() if n == 0]

    print()
    print(f"Clips read      : {len(per_clip)}")
    print(f"Boxes total     : {len(boxes_src)}")
    if empty:
        print(f"Clips with no boxes: {len(empty)}")

    # ---- transform into model space --------------------------------
    scale, pad_x, pad_y = letterbox_params(SRC_W, SRC_H, DST)

    print()
    print(f"Letterbox       : {SRC_W}x{SRC_H} -> {DST}x{DST}")
    print(f"  scale         : {scale:.4f}")
    print(f"  padding       : {pad_x:.1f} px left/right, {pad_y:.1f} px top/bottom")

    # Widths and heights scale; padding shifts position but not size.
    boxes = boxes_src * scale

    # ---- size distribution -----------------------------------------
    w, h = boxes[:, 0], boxes[:, 1]
    diag = np.sqrt(w * h)          # side of an equivalent square
    cell = DST / GRID              # 32 px

    print()
    print("Box size in model space (pixels):")
    for name, arr in (("width", w), ("height", h)):
        print(f"  {name:6s} min {arr.min():6.1f}  "
              f"p25 {np.percentile(arr, 25):6.1f}  "
              f"median {np.median(arr):6.1f}  "
              f"p75 {np.percentile(arr, 75):6.1f}  "
              f"max {arr.max():6.1f}")

    print()
    print(f"Grid cell is {cell:.0f} px. Equivalent square side vs cell:")
    for pct in (5, 25, 50, 75, 95):
        v = np.percentile(diag, pct)
        print(f"  p{pct:<3d} {v:6.1f} px  = {v / cell:5.2f} cells")

    tiny = (diag < 8).mean() * 100
    small = (diag < 16).mean() * 100
    print()
    print(f"Boxes under  8 px: {tiny:5.1f}%")
    print(f"Boxes under 16 px: {small:5.1f}%")

    # ---- aspect ratio ----------------------------------------------
    ar = w / h
    print()
    print(f"Aspect ratio (w/h): median {np.median(ar):.2f}, "
          f"p5 {np.percentile(ar, 5):.2f}, p95 {np.percentile(ar, 95):.2f}")

    # ---- anchors ----------------------------------------------------
    print()
    print("=" * 60)
    print(f"Clustering into {args.k} anchors (IoU k-means)...")

    clusters, assign, mean_iou = kmeans_iou(boxes, args.k, seed=args.seed)

    order = np.argsort(clusters[:, 0] * clusters[:, 1])
    clusters = clusters[order]

    print()
    print("Anchors in 224x224 pixel space:")
    for i, (cw, ch) in enumerate(clusters):
        share = (assign == order[i]).mean() * 100
        print(f"  anchor {i}: {cw:6.2f} x {ch:6.2f} px   ({share:4.1f}% of boxes)")

    print()
    print(f"Mean IoU between boxes and their anchor: {mean_iou:.3f}")

    if mean_iou < 0.5:
        print("  Low. The boxes vary a lot, or k is too small.")
    elif mean_iou < 0.65:
        print("  Reasonable for a small-object dataset.")
    else:
        print("  Good coverage.")

    # akida_models expects anchors in GRID CELL units, not pixels.
    print()
    print("Anchors in grid-cell units (what akida_models expects):")
    cell_anchors = clusters / cell
    print("ANCHORS = [")
    for cw, ch in cell_anchors:
        print(f"    [{cw:.4f}, {ch:.4f}],")
    print("]")

    print()
    print("=" * 60)
    print("Record these with the dataset. Training targets and inference")
    print("decoding must both use exactly these numbers.")


if __name__ == "__main__":
    main()