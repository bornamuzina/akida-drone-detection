"""
Is the model's box error systematic, or just noise?

Why this exists
---------------
The error analysis found the model puts its box on the real drone at a
median IoU of 0.40, and the jitter check showed the labels themselves
are steady to well under a pixel. So the looseness is the model's. The
question left is what kind of looseness.

If the boxes are scattered around the right answer, the only fix is
training -- a better loss, a finer grid, more resolution.

But if they are wrong the SAME WAY every time -- always 20% too wide,
always a pixel high -- that is a calibration offset, and calibration
offsets are corrected with arithmetic rather than a GPU.

What is measured
----------------
Every detection that clearly corresponds to a real drone (IoU >= 0.2,
so a detection sitting on something else is not dragged in) is compared
with its label:

    width ratio    predicted w / true w
    height ratio   predicted h / true h
    x offset       (predicted cx - true cx) / predicted size
    y offset       same, vertically

Offsets are divided by the PREDICTED size, not the true one, because at
correction time the true size is exactly what is unknown.

Then the correction is applied and the whole split is re-scored, so the
output says what the fix is worth rather than only that a bias exists.

Honesty about the numbers
-------------------------
Measuring the bias and testing the correction on the same split flatters
it -- the correction was fitted to that data. So: fit on validation,
then apply those fixed numbers to test with --scale_w and friends. The
script prints the command to do that.

Usage
-----
    python box_bias.py --run yolov2_s16_rgb_256 --split validation \\
        --weights best.weights.k2.h5

    python box_bias.py --run yolov2_s16_rgb_256 --split test \\
        --weights best.weights.k2.h5 \\
        --scale_w 1.23 --scale_h 1.19 --shift_x 0.01 --shift_y -0.04
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
import train as trainlib
import evaluate as ev


MATCH_FLOOR = 0.2       # below this a detection is not the same object


def correct(box, scale_w, scale_h, shift_x, shift_y):
    """
    Undo a systematic error on one (x, y, w, h) box.

    The shift is applied in units of the predicted box size, then the
    box is rescaled about its own centre. Order matters: rescaling
    first would change the size the shift is measured in.
    """
    x, y, w, h = box

    size = np.sqrt(max(w * h, 1e-6))

    cx = x + w / 2 - shift_x * size
    cy = y + h / 2 - shift_y * size

    w = w / scale_w
    h = h / scale_h

    return (cx - w / 2, cy - h / 2, w, h)


def score(frames, iou_thr, scale_w=1.0, scale_h=1.0, shift_x=0.0,
          shift_y=0.0):
    """
    Re-run the greedy matching over stored per-frame detections.

    frames is a list of (dets, truth). Keeping the detections lets the
    correction be tried at several settings without running the model
    again -- the forward pass is the slow part.
    """
    tp = fp = fn = 0
    matched_ious = []

    for dets, truth in frames:
        taken = set()

        for det in sorted(dets, key=lambda d: -d[4]):
            box = correct(det[:4], scale_w, scale_h, shift_x, shift_y)

            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(truth):
                if j in taken:
                    continue
                i = ev.iou(box, gt)
                if i > best_iou:
                    best_iou, best_j = i, j

            if best_iou >= iou_thr and best_j >= 0:
                taken.add(best_j)
                tp += 1
                matched_ious.append(best_iou)
            else:
                fp += 1

        fn += len(truth) - len(taken)

    return tp, fp, fn, matched_ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--weights", default="best.weights.h5")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--scale_w", type=float, default=None,
                    help="apply this instead of the measured value")
    ap.add_argument("--scale_h", type=float, default=None)
    ap.add_argument("--shift_x", type=float, default=None)
    ap.add_argument("--shift_y", type=float, default=None)
    args = ap.parse_args()

    run_dir = Path(config.RUNS_DIR) / args.run
    weights = run_dir / args.weights

    if not weights.exists():
        raise SystemExit(f"No weights at {weights}")

    given = args.scale_w is not None

    print()
    print(f"run    : {args.run}")
    print(f"split  : {args.split}")
    print(f"mode   : {'applying supplied correction' if given else 'measuring'}")
    print()

    from yolo_stride16 import yolo_base_stride16

    model = yolo_base_stride16(
        input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),
        classes=config.N_CLASSES,
        nb_box=config.N_ANCHORS,
        alpha=0.5,
    )
    model.load_weights(str(weights))

    splits = dataset.load_splits()
    ds = dataset.EventDataset(splits[args.split])

    frames = []
    ratios_w, ratios_h, offs_x, offs_y = [], [], [], []

    seen = 0

    for images, boxes_list in dataset.batches(ds, batch_size=32,
                                              shuffle=False):
        x = trainlib.to_uint8(images)
        preds = model(x, training=False).numpy()

        for p, truth in zip(preds, boxes_list):
            dets = ev.nms(ev.decode_predictions(p, conf_threshold=args.conf),
                          threshold=args.nms)

            frames.append((dets, [np.asarray(t, dtype=float) for t in truth]))

            for det in dets:
                bx, by, bw, bh = det[:4]

                best_iou, best = 0.0, None
                for gt in truth:
                    i = ev.iou(det[:4], gt)
                    if i > best_iou:
                        best_iou, best = i, gt

                if best is None or best_iou < MATCH_FLOOR:
                    continue

                gx, gy, gw, gh = [float(v) for v in best]

                if gw <= 0 or gh <= 0:
                    continue

                size = np.sqrt(max(bw * bh, 1e-6))

                ratios_w.append(bw / gw)
                ratios_h.append(bh / gh)
                offs_x.append(((bx + bw / 2) - (gx + gw / 2)) / size)
                offs_y.append(((by + bh / 2) - (gy + gh / 2)) / size)

        seen += len(images)
        if seen % 640 < 32:
            print(f"  {seen}/{len(ds)}")

    if not ratios_w:
        raise SystemExit("No detections overlapped a label. Nothing to measure.")

    rw = np.array(ratios_w)
    rh = np.array(ratios_h)
    ox = np.array(offs_x)
    oy = np.array(offs_y)

    print()
    print("=" * 66)
    print(f"SYSTEMATIC ERROR   ({len(rw)} detections on real drones)")
    print()
    print(f"  {'':<14}{'median':>10}{'mean':>10}{'p25':>10}{'p75':>10}")
    for name, a in (("width ratio", rw), ("height ratio", rh),
                    ("x offset", ox), ("y offset", oy)):
        print(f"  {name:<14}{np.median(a):>10.3f}{a.mean():>10.3f}"
              f"{np.percentile(a, 25):>10.3f}{np.percentile(a, 75):>10.3f}")

    print()
    print("  Width and height ratios are 1.0 if the model sizes boxes")
    print("  correctly. Offsets are 0.0 if it centres them correctly,")
    print("  and are in units of the box's own size -- 0.1 means a")
    print("  tenth of a box off.")

    # ---- baseline ---------------------------------------------------
    tp0, fp0, fn0, ious0 = score(frames, args.iou)

    print()
    print("=" * 66)
    print("WHAT A CORRECTION IS WORTH")
    print()

    if given:
        sw, sh, sx, sy = (args.scale_w, args.scale_h,
                          args.shift_x or 0.0, args.shift_y or 0.0)
        print("  Using the values supplied on the command line, which is")
        print("  the honest test if they were measured on another split.")
    else:
        sw, sh, sx, sy = (float(np.median(rw)), float(np.median(rh)),
                          float(np.median(ox)), float(np.median(oy)))
        print("  Using the medians measured above. These were fitted to")
        print("  THIS split, so the gain below is optimistic -- rerun on")
        print("  the other split with the command at the bottom.")

    print()
    print(f"  scale_w {sw:.4f}   scale_h {sh:.4f}   "
          f"shift_x {sx:.4f}   shift_y {sy:.4f}")

    tp1, fp1, fn1, ious1 = score(frames, args.iou, sw, sh, sx, sy)

    def line(tag, tp, fp, fn, ious):
        n_gt = tp + fn
        n_det = tp + fp
        rec = tp / n_gt if n_gt else 0.0
        prec = tp / n_det if n_det else 0.0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
        med = np.median(ious) if len(ious) else float("nan")
        print(f"  {tag:<12}{tp:>7}{fp:>7}{fn:>7}"
              f"{rec:>9.3f}{prec:>9.3f}{f1:>9.3f}{med:>10.3f}")

    print()
    print(f"  {'':<12}{'TP':>7}{'FP':>7}{'FN':>7}"
          f"{'recall':>9}{'prec':>9}{'F1':>9}{'med IoU':>10}")
    line("before", tp0, fp0, fn0, ious0)
    line("after", tp1, fp1, fn1, ious1)

    print()
    if tp1 > tp0:
        print(f"  {tp1 - tp0} detections cross the {args.iou} threshold that")
        print("  did not before, from arithmetic alone -- no retraining.")
    elif tp1 == tp0:
        print("  No change. The error is not systematic, so it cannot be")
        print("  corrected this way. A better loss or a finer grid is the")
        print("  route, not calibration.")
    else:
        print(f"  {tp0 - tp1} FEWER detections pass. The correction hurts,")
        print("  which means the median is not describing the error well.")

    if not given:
        other = "test" if args.split != "test" else "validation"
        print()
        print("  Confirm out of sample:")
        print(f"    python box_bias.py --run {args.run} --split {other} \\")
        print(f"      --weights {args.weights} \\")
        print(f"      --scale_w {sw:.4f} --scale_h {sh:.4f} "
              f"--shift_x {sx:.4f} --shift_y {sy:.4f}")


if __name__ == "__main__":
    main()