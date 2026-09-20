r"""
Where does the model actually get it wrong?

Counts alone do not help. "1,296 false positives" says nothing about
whether they are 1,296 different mistakes or one chimney seen 1,296
times -- and those need opposite fixes. The first wants more varied
data; the second wants one clip relabelled or a hard negative added.

So this writes three things:

  1. crops, with context, of every error, sorted worst-first
  2. contact sheets, so a few hundred can be reviewed in one go
  3. a per-clip table, plus a check for false positives that recur at
     the same pixel location across frames

The third is the point. A false positive that fires at (x, y) in frame
12 and again at (x, y) in frames 13 through 200 is a fixed background
object, not noise. The script clusters FP centres within each clip and
reports any cluster spanning many frames.

Crops carry context deliberately. A 12 px drone cropped to 12 px tells
you nothing; the question is always what is AROUND it.

Usage:
    python error_analysis.py --run yolov2_s16_rgb_256 --split validation
    python error_analysis.py --run yolov2_s16_rgb_256 --split test
"""

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json

import cv2
import numpy as np

# This script lives one level down from the model code it imports, so
# put the parent directory on the path before importing any of it.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import dataset
import train as trainlib
import evaluate as ev


# How much context to keep around each error. The drone is ~12 px, so
# this shows roughly eight drone-widths of surroundings.
CROP = 96

# Contact sheet geometry.
SHEET_COLS = 10
SHEET_ROWS = 8
TILE = 112                      # CROP plus room for a label strip


# ============================================================
# CROPPING
# ============================================================

def crop_with_context(img, cx, cy, size=CROP):
    """
    Square crop centred on (cx, cy), clamped to the frame.

    Returns the crop and the offset used, so a box can be redrawn in
    crop coordinates afterwards.
    """
    h, w = img.shape[:2]

    if w < size or h < size:
        return img.copy(), (0, 0)

    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half

    # Clamp rather than pad: a black border would look like image
    # content and mislead whoever is reviewing these.
    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))

    return img[y0:y0 + size, x0:x0 + size].copy(), (x0, y0)


def magnify(crop, zoom):
    """
    Blow the crop up with nearest-neighbour.

    The targets are ten pixels across. At native size two boxes two
    pixels apart are indistinguishable, and the second one drawn simply
    paints over the first. Nearest-neighbour rather than a smooth
    interpolation so the pixel grid stays visible -- this is for
    judging alignment, not for looking nice.
    """
    if zoom <= 1:
        return crop

    return cv2.resize(crop, None, fx=zoom, fy=zoom,
                      interpolation=cv2.INTER_NEAREST)


def draw_box(crop, box, offset, colour, thickness=1, zoom=1):
    x, y, bw, bh = box
    ox, oy = offset

    x0 = int(round((x - ox) * zoom))
    y0 = int(round((y - oy) * zoom))
    x1 = int(round((x - ox + bw) * zoom))
    y1 = int(round((y - oy + bh) * zoom))

    cv2.rectangle(crop, (x0, y0), (x1, y1), colour, thickness)


# ============================================================
# CONTACT SHEETS
# ============================================================

def contact_sheets(crops, labels, out_dir, prefix):
    """
    Tile crops into browsable sheets.

    Nobody will open 800 separate files. Eight sheets they will.
    """
    if not crops:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # The tile has to fit whatever the crops actually are -- they get
    # magnified now, so a fixed 112 would crop them or run off the
    # sheet. Fewer columns too, or a magnified sheet is unopenable.
    ch0, cw0 = crops[0].shape[:2]
    tile = max(ch0, cw0) + 16

    cols = max(1, min(SHEET_COLS, 2400 // tile))
    rows = max(1, min(SHEET_ROWS, 1800 // tile))

    per_sheet = cols * rows
    n_sheets = (len(crops) + per_sheet - 1) // per_sheet

    for s in range(n_sheets):
        sheet = np.full((rows * tile, cols * tile, 3),
                        30, dtype=np.uint8)

        for i in range(per_sheet):
            idx = s * per_sheet + i
            if idx >= len(crops):
                break

            r, c = divmod(i, cols)
            y0, x0 = r * tile, c * tile

            crop = crops[idx]
            ch, cw = crop.shape[:2]
            sheet[y0:y0 + ch, x0:x0 + cw] = crop

            cv2.putText(sheet, labels[idx][:18],
                        (x0 + 2, y0 + tile - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)

        cv2.imwrite(str(out_dir / f"{prefix}_sheet_{s:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    return n_sheets


# ============================================================
# RECURRING FALSE POSITIVES
# ============================================================

def find_recurring(fps, radius=12, min_frames=5):
    """
    Group one clip's false positives by location.

    A cluster spanning many frames is a fixed object the model keeps
    calling a drone -- a chimney, an antenna, a mark on the lens. That
    is a different problem from scattered one-off errors, and the fix
    is different too.

    Greedy clustering is enough: the question is whether tight clusters
    exist at all, not their exact membership.
    """
    clusters = []

    for fp in sorted(fps, key=lambda f: -f["conf"]):
        cx, cy = fp["cx"], fp["cy"]

        for cl in clusters:
            if abs(cx - cl["cx"]) <= radius and abs(cy - cl["cy"]) <= radius:
                cl["members"].append(fp)
                n = len(cl["members"])
                # Running mean keeps the centre from drifting to
                # whichever point happened to be added last.
                cl["cx"] += (cx - cl["cx"]) / n
                cl["cy"] += (cy - cl["cy"]) / n
                break
        else:
            clusters.append({"cx": cx, "cy": cy, "members": [fp]})

    out = []
    for cl in clusters:
        frames = {m["frame"] for m in cl["members"]}
        if len(frames) >= min_frames:
            confs = [m["conf"] for m in cl["members"]]
            out.append({
                "cx": cl["cx"], "cy": cl["cy"],
                "n_detections": len(cl["members"]),
                "n_frames": len(frames),
                "mean_conf": float(np.mean(confs)),
                "max_conf": float(max(confs)),
                "example": cl["members"][0],
            })

    return sorted(out, key=lambda c: -c["n_frames"])


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--weights", default="best.weights.h5",
                    help="file name inside the run folder")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--window", type=int, default=48,
                    help="pixels of context around the box, before zoom")
    ap.add_argument("--zoom", type=int, default=6,
                    help="magnification; 1 writes crops at native size")
    ap.add_argument("--max_crops", type=int, default=400,
                    help="crops written per category, worst first")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(config.RUNS_DIR) / args.run
    weights = run_dir / args.weights
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}")

    out_dir = Path(args.out) if args.out else run_dir / f"errors_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"run    : {args.run}")
    print(f"split  : {args.split}")
    print(f"conf   : {args.conf}   IoU {args.iou}   NMS {args.nms}")
    print(f"output : {out_dir}")
    print()

    # ---- model ------------------------------------------------------
    from yolo_stride16 import yolo_base_stride16

    model = yolo_base_stride16(
        input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),
        classes=config.N_CLASSES,
        nb_box=config.N_ANCHORS,
        alpha=0.5,
    )
    model.load_weights(str(weights))

    # ---- data -------------------------------------------------------
    splits = dataset.load_splits()
    ds = dataset.EventDataset(splits[args.split])

    # dataset.batches does not report which clip a sample came from, so
    # recover it from the flat index the dataset was built with. If the
    # attribute is not there, fall back to a running counter -- the
    # per-clip table is then useless but the crops still work.
    index = getattr(ds, "index", None)
    if index is None:
        print("  NOTE: dataset has no .index, per-clip attribution off")

    fp_records = []
    fn_records = []
    per_clip = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0})

    sample_i = 0
    n_tp = 0

    for images, boxes_list in dataset.batches(ds, batch_size=32, shuffle=False):
        x = trainlib.to_uint8(images)
        preds = model(x, training=False).numpy()

        for img_f, p, truth in zip(images, preds, boxes_list):
            clip, frame = "unknown", sample_i
            if index is not None and sample_i < len(index):
                entry = index[sample_i]
                clip = entry[0]
                frame = entry[1] if len(entry) > 1 else sample_i
            sample_i += 1

            img = np.clip(np.asarray(img_f), 0, 255).astype(np.uint8)

            dets = ev.decode_predictions(p, conf_threshold=args.conf)
            dets = ev.nms(dets, threshold=args.nms)

            per_clip[clip]["gt"] += len(truth)

            # Where this frame's false positives start, so they can be
            # labelled once the frame is fully matched and it is known
            # whether the drone was found at all.
            fp_start = len(fp_records)

            matched = set()
            for det in sorted(dets, key=lambda d: -d[4]):
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(truth):
                    if j in matched:
                        continue
                    i = ev.iou(det[:4], gt)
                    if i > best_iou:
                        best_iou, best_j = i, j

                if best_iou >= args.iou and best_j >= 0:
                    matched.add(best_j)
                    per_clip[clip]["tp"] += 1
                    n_tp += 1
                else:
                    x_, y_, w_, h_ = det[:4]
                    fp_records.append({
                        "clip": clip, "frame": int(frame),
                        "conf": float(det[4]),
                        "cx": x_ + w_ / 2, "cy": y_ + h_ / 2,
                        "w": w_, "h": h_,
                        "best_iou": float(best_iou),
                        "img": img, "box": det[:4],
                        "gt_box": (np.asarray(truth[best_j], dtype=float)
                                   if best_j >= 0 else None),
                    })
                    per_clip[clip]["fp"] += 1

            for j, gt in enumerate(truth):
                if j in matched:
                    continue
                x_, y_, w_, h_ = gt
                fn_records.append({
                    "clip": clip, "frame": int(frame),
                    "cx": x_ + w_ / 2, "cy": y_ + h_ / 2,
                    "w": w_, "h": h_,
                    "size": float(np.sqrt(w_ * h_)),
                    "img": img, "box": gt,
                })
                per_clip[clip]["fn"] += 1

            # ---- why was each of this frame's FPs an FP? ------------
            # With one ground-truth box per frame the cases are clean:
            # either the detection overlapped the real drone but not
            # enough to count, or it did not overlap it at all -- and
            # in that case it matters whether the drone was found by
            # some other detection (a duplicate) or missed entirely
            # (the model was looking somewhere else).
            frame_tp = len(matched)
            frame_miss = len(truth) - len(matched)

            for r in fp_records[fp_start:]:
                if r["best_iou"] > 0:
                    r["kind"] = "near_miss"
                elif frame_tp > 0:
                    r["kind"] = "duplicate"
                elif frame_miss > 0:
                    r["kind"] = "elsewhere"
                else:
                    r["kind"] = "no_target"

        if sample_i % 640 < 32:
            print(f"  {sample_i}/{len(ds)}")

    print()
    print(f"true positives  : {n_tp}")
    print(f"false positives : {len(fp_records)}")
    print(f"false negatives : {len(fn_records)}")

    # ---- crops ------------------------------------------------------
    # Worst first: highest-confidence FPs, and largest missed drones,
    # since a missed 25 px drone is a more surprising failure than a
    # missed 7 px one.
    fp_records.sort(key=lambda r: -r["conf"])
    fn_records.sort(key=lambda r: -r["size"])

    fp_dir = out_dir / "false_positives"
    fn_dir = out_dir / "false_negatives"
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    fp_crops, fp_labels = [], []
    fn_crops, fn_labels = [], []

    for i, r in enumerate(fp_records[:args.max_crops]):
        crop, off = crop_with_context(r["img"], r["cx"], r["cy"],
                                      size=args.window)
        crop = magnify(crop, args.zoom)
        # The real drone goes down first and the claimed box on top, so
        # that where the two nearly coincide it is the model's box that
        # survives -- the green underneath still shows at the edges.
        if r.get("gt_box") is not None:
            draw_box(crop, r["gt_box"], off, (0, 255, 0),  # green: real
                     zoom=args.zoom)
        draw_box(crop, r["box"], off, (0, 0, 255),         # red: claimed
                 zoom=args.zoom)
        cv2.imwrite(str(fp_dir /
                        f"{i:04d}_{r.get('kind', 'fp')}_{r['conf']:.2f}"
                        f"_{r['clip']}_{r['frame']:05d}.jpg"),
                    crop)
        fp_crops.append(crop)
        fp_labels.append(f"{r.get('kind', '')[:4]} {r['conf']:.2f}")

    for i, r in enumerate(fn_records[:args.max_crops]):
        crop, off = crop_with_context(r["img"], r["cx"], r["cy"],
                                      size=args.window)
        crop = magnify(crop, args.zoom)
        draw_box(crop, r["box"], off, (0, 255, 0),         # green: missed
                 zoom=args.zoom)
        cv2.imwrite(str(fn_dir /
                        f"{i:04d}_{r['size']:.0f}px_{r['clip']}_{r['frame']:05d}.jpg"),
                    crop)
        fn_crops.append(crop)
        fn_labels.append(f"{r['size']:.0f}px {str(r['clip'])[-7:]}")

    n_fp_sheets = contact_sheets(fp_crops, fp_labels, out_dir, "fp")
    n_fn_sheets = contact_sheets(fn_crops, fn_labels, out_dir, "fn")

    print()
    print(f"wrote {len(fp_crops)} FP crops, {n_fp_sheets} sheets")
    print(f"wrote {len(fn_crops)} FN crops, {n_fn_sheets} sheets")

    # ---- recurring false positives ----------------------------------
    by_clip = defaultdict(list)
    for r in fp_records:
        by_clip[r["clip"]].append(r)

    recurring = []
    for clip, fps in by_clip.items():
        for c in find_recurring(fps):
            c["clip"] = clip
            recurring.append(c)

    recurring.sort(key=lambda c: -c["n_frames"])

    print()
    print("=" * 66)
    print("RECURRING FALSE POSITIVES")
    print()
    print("  A cluster spanning many frames is a fixed object the model")
    print("  keeps calling a drone, not scattered noise.")
    print()

    if recurring:
        covered = sum(c["n_detections"] for c in recurring)
        print(f"  {len(recurring)} clusters, covering {covered} of "
              f"{len(fp_records)} false positives "
              f"({covered / max(len(fp_records), 1) * 100:.0f}%)")
        print()
        print(f"  {'clip':<16} {'at':>12} {'frames':>7} {'dets':>6} {'conf':>6}")
        for c in recurring[:20]:
            at = f"({c['cx']:.0f},{c['cy']:.0f})"
            print(f"  {str(c['clip']):<16} {at:>12} {c['n_frames']:>7} "
                  f"{c['n_detections']:>6} {c['mean_conf']:>6.2f}")

        # A crop of each, so the object can be identified by eye.
        rec_dir = out_dir / "recurring"
        rec_dir.mkdir(exist_ok=True)
        for i, c in enumerate(recurring[:40]):
            ex = c["example"]
            crop, off = crop_with_context(ex["img"], c["cx"], c["cy"],
                                          size=args.window)
            crop = magnify(crop, args.zoom)
            if ex.get("gt_box") is not None:
                draw_box(crop, ex["gt_box"], off, (0, 255, 0),
                         zoom=args.zoom)
            draw_box(crop, ex["box"], off, (0, 0, 255), zoom=args.zoom)
            cv2.imwrite(str(rec_dir /
                            f"{i:03d}_{c['n_frames']:04d}f_{c['clip']}.jpg"),
                        crop)
    else:
        print("  None found. The false positives are scattered, which")
        print("  points at general confusion rather than specific objects.")

    # ---- where in the frame -----------------------------------------
    print()
    print("=" * 66)
    print("POSITION IN FRAME")
    print()
    S = config.INPUT_SIZE
    pad = config.PAD_Y
    band_h = (S - 2 * pad) / 3

    def band(cy):
        if cy < pad + band_h:
            return "upper (sky)"
        if cy < pad + 2 * band_h:
            return "middle"
        return "lower (ground)"

    fp_bands = defaultdict(int)
    fn_bands = defaultdict(int)
    for r in fp_records:
        fp_bands[band(r["cy"])] += 1
    for r in fn_records:
        fn_bands[band(r["cy"])] += 1

    print(f"  {'band':<16} {'FP':>8} {'FN':>8}")
    for b in ("upper (sky)", "middle", "lower (ground)"):
        print(f"  {b:<16} {fp_bands[b]:>8} {fn_bands[b]:>8}")

    # ---- per clip ---------------------------------------------------
    print()
    print("=" * 66)
    print("WORST CLIPS  (by false negatives)")
    print()
    rows = []
    for clip, c in per_clip.items():
        rec = c["tp"] / max(c["gt"], 1)
        prec = c["tp"] / max(c["tp"] + c["fp"], 1)
        rows.append((str(clip), c["gt"], c["tp"], c["fp"], c["fn"], rec, prec))

    rows.sort(key=lambda r: -r[4])

    print(f"  {'clip':<16} {'gt':>5} {'tp':>5} {'fp':>5} {'fn':>5} "
          f"{'rec':>6} {'prec':>6}")
    for r in rows[:15]:
        print(f"  {r[0]:<16} {r[1]:>5} {r[2]:>5} {r[3]:>5} {r[4]:>5} "
              f"{r[5]:>6.3f} {r[6]:>6.3f}")

    # ---- files ------------------------------------------------------
    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "gt", "tp", "fp", "fn", "recall", "precision"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4],
                        f"{r[5]:.4f}", f"{r[6]:.4f}"])

    with open(out_dir / "false_positives.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "frame", "kind", "conf",
                    "cx", "cy", "w", "h", "best_iou"])
        for r in fp_records:
            w.writerow([r["clip"], r["frame"], r.get("kind", ""),
                        f"{r['conf']:.4f}",
                        f"{r['cx']:.1f}", f"{r['cy']:.1f}",
                        f"{r['w']:.1f}", f"{r['h']:.1f}",
                        f"{r['best_iou']:.3f}"])

    with open(out_dir / "false_negatives.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "frame", "cx", "cy", "w", "h", "size_px"])
        for r in fn_records:
            w.writerow([r["clip"], r["frame"],
                        f"{r['cx']:.1f}", f"{r['cy']:.1f}",
                        f"{r['w']:.1f}", f"{r['h']:.1f}",
                        f"{r['size']:.1f}"])

    with open(out_dir / "recurring.json", "w") as f:
        # numpy scalars come out of the decode and json refuses them,
        # so hand it a plain-Python fallback rather than converting
        # every field by hand.
        json.dump([{k: v for k, v in c.items() if k != "example"}
                   for c in recurring], f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- what kind of error is each false positive? -----------------
    print()
    print("=" * 66)
    print("WHAT THE FALSE POSITIVES ACTUALLY ARE")
    print()

    kinds = defaultdict(int)
    for r in fp_records:
        kinds[r.get("kind", "no_target")] += 1

    total_fp = max(len(fp_records), 1)

    labels = {
        "near_miss": "overlapped the drone, under the IoU threshold",
        "duplicate": "second box on a drone already found",
        "elsewhere": "drone in frame, model fired somewhere else",
        "no_target": "nothing to find in the frame",
    }

    for k in ("near_miss", "duplicate", "elsewhere", "no_target"):
        n = kinds.get(k, 0)
        print(f"  {k:<12} {n:>6}  {n / total_fp * 100:5.1f}%  {labels[k]}")

    print()

    near = [r["best_iou"] for r in fp_records if r.get("kind") == "near_miss"]
    if near:
        near = np.array(near)
        print(f"  IoU of the near misses: median {np.median(near):.3f}, "
              f"p25 {np.percentile(near, 25):.3f}, "
              f"p75 {np.percentile(near, 75):.3f}")
        for t in (0.3, 0.4, 0.45):
            share = float((near >= t).mean())
            print(f"    {share * 100:5.1f}% would count at IoU {t}")
        print()

    double_counted = kinds.get("near_miss", 0)
    print(f"  {double_counted} of the {len(fp_records)} false positives are")
    print(f"  paired with one of the {len(fn_records)} false negatives --")
    print("  the same detection penalised twice.")
    print()
    print("  near_miss and duplicate are box-accuracy problems and no")
    print("  amount of extra data fixes them. Only 'elsewhere' and")
    print("  'no_target' are the model confusing something else for a")
    print("  drone, and those are what a synthetic dataset can address.")

    print()
    print("=" * 66)
    print("Start with the contact sheets. The per-clip table says which")
    print("clips to distrust; the recurring table says whether the same")
    print("object is being mistaken over and over.")


if __name__ == "__main__":
    main()