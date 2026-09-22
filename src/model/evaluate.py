"""
Evaluate a trained detector properly.

Training reported a crude recall: did the correct slot fire above 0.5.
That ignores two things that matter -- whether the predicted box is
actually on the drone, and how many false positives the model produces
to achieve its hits. A model that fires everywhere would score well on
that metric and be useless.

This decodes predictions into real boxes, applies NMS, matches against
ground truth by IoU, and reports precision, recall and average
precision across the full confidence range.

It also breaks results down by target size, which for this dataset is
the finding that matters: 91% of boxes are under 16 px and 16% under
8 px, and an averaged number would hide whether the small ones work at
all.

Negatives
---------
--negatives adds the airplane, bird and helicopter clips, whose frames
contain no drone. Every detection on them is a false positive, so the
score drops -- that is the point. It is the first measurement of the
false-alarm rate, which the drone-only splits cannot produce because
every one of their frames contains a drone.

The two numbers are not comparable, so they are written to different
files: eval_test.json and eval_test_neg.json. Quote both. A drop from
one to the other is a stricter measurement, not a worse model; a drop
in the FIRST one between runs is a worse model.

Usage:
    python evaluate.py --run full_20ms --split validation
    python evaluate.py --run full_20ms --split test        # once only
    python evaluate.py --run full_20ms --split test --negatives
"""

from pathlib import Path
import argparse
import json

import numpy as np
import tensorflow as tf

import config
import dataset
import train as trainlib


# ============================================================
# DECODING
# ============================================================

def decode_predictions(pred, conf_threshold=0.01):
    """
    Raw model output -> list of (x, y, w, h, score) in 224-space pixels.

    The inverse of what the loss does internally: sigmoid on centre and
    objectness, exponential on size relative to the anchor. These must
    match the loss exactly or the boxes come out wrong.
    """
    grid = config.GRID
    n_anchors = config.N_ANCHORS
    n_classes = config.N_CLASSES
    cell = config.CELL

    anchors = np.array(config.ANCHORS, dtype=np.float32)

    pred = np.reshape(pred, (grid, grid, n_anchors, 5 + n_classes))

    boxes = []

    for row in range(grid):
        for col in range(grid):
            for a in range(n_anchors):
                p = pred[row, col, a]

                obj = 1.0 / (1.0 + np.exp(-p[4]))
                if obj < conf_threshold:
                    continue

                cls = 1.0 / (1.0 + np.exp(-p[5]))
                score = obj * cls

                cx = (1.0 / (1.0 + np.exp(-p[0])) + col) * cell
                cy = (1.0 / (1.0 + np.exp(-p[1])) + row) * cell
                w = np.exp(np.clip(p[2], -8, 8)) * anchors[a, 0] * cell
                h = np.exp(np.clip(p[3], -8, 8)) * anchors[a, 1] * cell

                boxes.append((cx - w / 2, cy - h / 2, w, h, float(score)))

    return boxes


def iou(a, b):
    """IoU between two (x, y, w, h) boxes."""
    ax0, ay0, aw, ah = a[:4]
    bx0, by0, bw, bh = b[:4]

    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)

    inter = iw * ih
    union = aw * ah + bw * bh - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, threshold=0.4):
    """
    Suppress overlapping detections, keeping the most confident.

    Without this, the same drone is reported several times -- once per
    anchor that fired -- and every duplicate counts as a false positive.
    """
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []

    for box in boxes:
        if all(iou(box, k) < threshold for k in kept):
            kept.append(box)

    return kept


# ============================================================
# MATCHING
# ============================================================

def match(preds, truths, iou_threshold=0.5):
    """
    Greedy matching, highest-confidence prediction first.

    Returns (matches, n_truth) where matches is a list of
    (score, is_true_positive) and each ground-truth box can be claimed
    at most once. A second prediction on an already-matched drone is a
    false positive, which is the correct treatment.

    With no ground truth at all -- a negative frame -- every prediction
    is a false positive, which is also correct.
    """
    preds = sorted(preds, key=lambda b: -b[4])
    claimed = [False] * len(truths)

    out = []

    for p in preds:
        best_i = -1
        best_iou = iou_threshold

        for i, t in enumerate(truths):
            if claimed[i]:
                continue
            v = iou(p, t)
            if v >= best_iou:
                best_iou = v
                best_i = i

        if best_i >= 0:
            claimed[best_i] = True
            out.append((p[4], True))
        else:
            out.append((p[4], False))

    return out, len(truths)


def average_precision(matches, n_truth):
    """
    Area under the precision-recall curve, computed over all confidence
    thresholds rather than one arbitrary cut.

    A single-threshold number can be tuned to look good; AP cannot.
    """
    if n_truth == 0:
        return 0.0, [], []

    matches = sorted(matches, key=lambda m: -m[0])

    tp = 0
    fp = 0

    precisions = []
    recalls = []

    for score, is_tp in matches:
        if is_tp:
            tp += 1
        else:
            fp += 1

        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_truth)

    if not precisions:
        return 0.0, [], []

    # Interpolate: precision at each recall is the best achievable at
    # that recall or higher. Standard for AP and stops noise in the
    # curve depressing the number.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    prev_r = 0.0

    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_r)
        prev_r = r

    return ap, precisions, recalls


# ============================================================
# MAIN
# ============================================================

SIZE_BINS = [(0, 8), (8, 12), (12, 16), (16, 999)]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--run", required=True, help="run name under RUNS_DIR")
    ap_.add_argument("--split", default="validation",
                     choices=["train", "validation", "test"])
    ap_.add_argument("--tensors", default=None)
    ap_.add_argument("--alpha", type=float, default=0.5)
    ap_.add_argument("--conf", type=float, default=0.5,
                     help="threshold for the headline precision/recall")
    ap_.add_argument("--iou", type=float, default=0.5)
    ap_.add_argument("--nms", type=float, default=0.4)
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--weights", default="best.weights.h5",
                     help="file name inside the run folder")
    ap_.add_argument("--negatives", action="store_true",
                     help="include the non-drone clips; measures the "
                          "false-alarm rate, and is not comparable with "
                          "the number from without them")
    args = ap_.parse_args()

    # from akida_models import yolo_base
    from yolo_stride16 import yolo_base_stride16

    run_dir = Path(config.RUNS_DIR) / args.run
    weights = run_dir / args.weights

    if not weights.exists():
        raise SystemExit(f"No weights at {weights}")

    tensor_dir = Path(args.tensors) if args.tensors else config.TENSOR_DIR

    # ---- data --------------------------------------------------------
    splits = dataset.load_splits()
    ds = dataset.EventDataset(
        dataset.clips_for(splits, args.split, args.negatives),
        tensor_dir=tensor_dir, verbose=False,
        keep_empty=args.negatives)

    n = len(ds) if not args.limit else min(args.limit, len(ds))

    print(f"run    : {args.run}")
    print(f"split  : {args.split}  ({n} samples)")
    print(f"conf   : {args.conf}   IoU {args.iou}   NMS {args.nms}")
    if args.negatives:
        print(f"neg    : {ds.n_negative} boxless frames included")
    print()

    # ---- model -------------------------------------------------------
    #model = yolo_base(input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),classes=config.N_CLASSES,nb_box=config.N_ANCHORS,alpha=args.alpha,)
    model = yolo_base_stride16(
        input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),
        classes=config.N_CLASSES,
        nb_box=config.N_ANCHORS,
        alpha=args.alpha,
    )
    model.load_weights(str(weights))

    # ---- run ---------------------------------------------------------
    all_matches = []
    total_truth = 0

    tp = fp = fn = 0

    # False positives landing on a frame that held no drone at all.
    # These are the false-alarm rate, and they are worth separating from
    # false positives on a frame where the model simply missed.
    fp_on_empty = 0
    n_empty = 0
    n_empty_with_detection = 0

    # Per-size-bin tallies, so the average does not hide the small ones.
    bin_truth = {b: 0 for b in SIZE_BINS}
    bin_hit = {b: 0 for b in SIZE_BINS}

    batch = 32

    for start in range(0, n, batch):
        idx = range(start, min(start + batch, n))

        images = []
        truths = []

        for i in idx:
            img, boxes, _ = ds[i]
            images.append(img)
            truths.append(boxes)

        x = trainlib.to_uint8(np.stack(images))
        preds = model.predict(x, verbose=0)

        for p, truth in zip(preds, truths):
            boxes = decode_predictions(p, conf_threshold=0.01)
            boxes = nms(boxes, threshold=args.nms)

            m, nt = match(boxes, truth, iou_threshold=args.iou)

            all_matches.extend(m)
            total_truth += nt

            # Headline numbers at the chosen confidence.
            kept = [b for b in boxes if b[4] >= args.conf]
            mk, _ = match(kept, truth, iou_threshold=args.iou)

            hits = sum(1 for _, t in mk if t)
            tp += hits
            fp += len(mk) - hits
            fn += nt - hits

            if nt == 0:
                n_empty += 1
                fp_on_empty += len(mk)
                if kept:
                    n_empty_with_detection += 1

            # Which sizes were found?
            matched_truth = set()
            for pred in sorted(kept, key=lambda b: -b[4]):
                for ti, t in enumerate(truth):
                    if ti in matched_truth:
                        continue
                    if iou(pred, t) >= args.iou:
                        matched_truth.add(ti)
                        break

            for ti, t in enumerate(truth):
                size = np.sqrt(t[2] * t[3])
                for b in SIZE_BINS:
                    if b[0] <= size < b[1]:
                        bin_truth[b] += 1
                        if ti in matched_truth:
                            bin_hit[b] += 1
                        break

        if start % (batch * 20) == 0:
            print(f"  {start}/{n}")

    # ---- report -------------------------------------------------------
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    ap, precisions, recalls = average_precision(all_matches, total_truth)

    print()
    print("=" * 56)
    print(f"AT CONFIDENCE {args.conf}")
    print(f"  true positives  : {tp}")
    print(f"  false positives : {fp}")
    print(f"  false negatives : {fn}")
    print(f"  precision       : {precision:.3f}")
    print(f"  recall          : {recall:.3f}")
    print(f"  F1              : {f1:.3f}")
    print()
    print(f"AVERAGE PRECISION (AP@{args.iou}) : {ap:.3f}")
    print("  Area under the precision-recall curve, over all thresholds.")

    # ---- false alarms --------------------------------------------------
    if n_empty:
        print()
        print("FALSE ALARMS  (frames with no drone in them)")
        print(f"  empty frames    : {n_empty}")
        print(f"  frames that fired: {n_empty_with_detection} "
              f"({n_empty_with_detection / n_empty * 100:.1f}%)")
        print(f"  detections      : {fp_on_empty} "
              f"({fp_on_empty / n_empty:.3f} per frame)")
        print()
        print("  This is the number the drone-only splits cannot produce.")
        print("  In deployment an empty frame is the common case, so the")
        print("  per-frame rate matters more than the AP above.")

    print()
    print("RECALL BY TARGET SIZE")
    print(f"  {'size (px)':>12}  {'boxes':>6}  {'found':>6}  {'recall':>7}")
    for b in SIZE_BINS:
        t = bin_truth[b]
        if t == 0:
            continue
        label = f"{b[0]}-{b[1]}" if b[1] < 999 else f"{b[0]}+"
        print(f"  {label:>12}  {t:6d}  {bin_hit[b]:6d}  "
              f"{bin_hit[b] / t:7.3f}")

    print()
    print("  The smallest bin is the one that matters. An averaged recall")
    print("  hides whether sub-8px targets work at all.")

    # ---- a few points on the curve ------------------------------------
    if precisions:
        print()
        print("PRECISION-RECALL CURVE")
        print(f"  {'recall':>8}  {'precision':>10}")
        for target in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            best = None
            for p, r in zip(precisions, recalls):
                if r >= target:
                    best = p
                    break
            if best is not None:
                print(f"  {target:8.1f}  {best:10.3f}")

    # ---- save ---------------------------------------------------------
    out = {
        "run": args.run,
        "split": args.split,
        "negatives": args.negatives,
        "n_samples": n,
        "n_empty_frames": n_empty,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "nms_threshold": args.nms,
        "tp": tp, "fp": fp, "fn": fn,
        "fp_on_empty_frames": fp_on_empty,
        "empty_frames_with_detection": n_empty_with_detection,
        "false_alarms_per_empty_frame": (
            fp_on_empty / n_empty if n_empty else None),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
        "recall_by_size": {
            f"{b[0]}-{b[1]}": {
                "boxes": bin_truth[b],
                "found": bin_hit[b],
                "recall": bin_hit[b] / bin_truth[b] if bin_truth[b] else None,
            }
            for b in SIZE_BINS
        },
    }

    # A run with negatives is a different measurement, so it gets its own
    # file. Overwriting eval_test.json with it would destroy the only
    # number comparable with every earlier run.
    suffix = "_neg" if args.negatives else ""
    path = run_dir / f"eval_{args.split}{suffix}.json"

    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()