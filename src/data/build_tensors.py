"""
Turn simulated events into model input tensors.

For each labelled video frame this takes the events in the window
immediately preceding it, accumulates them into a single signed
channel, letterboxes to 224x224, and writes the tensor together with
the boxes transformed by exactly the same geometry.

Representation: signed accumulation / diff frame.
  ON event  -> +1 at that pixel
  OFF event -> -1 at that pixel
A pixel therefore holds (ON count - OFF count) over the window.

Chosen because the Akida input layer accepts 1 or 3 channels, not 2 --
a 2-channel polarity histogram builds in Keras but fails cnn2snn
conversion with "Only signed inputs are supported".

Nothing is normalised or clipped here. Raw signed counts are written so
that normalisation stays a training-time choice and can be changed
without regenerating the dataset.

Usage:
    python build_tensors.py --dat <file.dat> --labels <file.csv>
                            --video <file.mp4> --output <dir>
                            [--window_ms 20] [--stride 1]
"""

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator


DST = 224


# ============================================================
# GEOMETRY
# ============================================================

def letterbox_params(src_w, src_h, dst=DST):
    """
    Scale factor and padding for fitting src into dst x dst without
    distorting the aspect ratio.

    Everything downstream -- tensor, boxes, anchors -- must use these
    same three numbers. Box misalignment almost always traces back to
    two parts of a pipeline disagreeing here.
    """
    scale = min(dst / src_w, dst / src_h)
    pad_x = (dst - src_w * scale) / 2.0
    pad_y = (dst - src_h * scale) / 2.0
    return scale, pad_x, pad_y


def transform_box(box, scale, pad_x, pad_y):
    """(x, y, w, h) in source pixels -> the same box in 224x224 space."""
    x, y, w, h = box
    return (x * scale + pad_x, y * scale + pad_y, w * scale, h * scale)


# ============================================================
# LABELS
# ============================================================

def load_labels_csv(path):
    boxes = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {"frame", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing column(s): {sorted(missing)}")

        for row in reader:
            boxes[int(float(row["frame"]))].append((
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            ))

    return boxes


# ============================================================
# EVENTS
# ============================================================

def load_all_events(dat_path):
    chunks = []
    for chunk in EventsIterator(input_path=str(dat_path), delta_t=100000):
        if len(chunk):
            chunks.append(chunk.copy())

    if not chunks:
        raise RuntimeError(f"No events in {dat_path}")

    return np.concatenate(chunks)


def accumulate_signed(events, src_w, src_h):
    """
    Signed accumulation at source resolution.

    np.add.at is used rather than plain indexing because several events
    land on the same pixel within a window and we want them summed, not
    overwritten. That count is the signal -- an edge hit forty times is
    not the same as one hit once.
    """
    frame = np.zeros((src_h, src_w), dtype=np.float32)

    if len(events) == 0:
        return frame

    x = events["x"].astype(np.int32)
    y = events["y"].astype(np.int32)
    p = events["p"]

    ok = (x >= 0) & (x < src_w) & (y >= 0) & (y < src_h)
    x, y, p = x[ok], y[ok], p[ok]

    signed = np.where(p > 0, 1.0, -1.0).astype(np.float32)
    np.add.at(frame, (y, x), signed)

    return frame


def letterbox_frame(frame, scale, pad_x, pad_y, dst=DST):
    """
    Resize by `scale` then pad into a dst x dst square.

    INTER_AREA is the right choice when shrinking: it averages the
    source pixels covering each output pixel. Nearest-neighbour would
    drop most events entirely at a scale of 0.35.

    Note that averaging changes the units -- the output is no longer an
    integer event count but a density. That is fine and is what we want,
    but it is worth knowing when interpreting values.
    """
    src_h, src_w = frame.shape
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((dst, dst), dtype=np.float32)
    y0 = int(round(pad_y))
    x0 = int(round(pad_x))
    out[y0:y0 + new_h, x0:x0 + new_w] = resized

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--video", required=True, help="source mp4, for fps and geometry")
    ap.add_argument("--output", required=True)
    ap.add_argument("--window_ms", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth labelled frame")
    ap.add_argument("--quiet_threshold", type=int, default=10,
                    help="frames whose boxes contain fewer than this many "
                         "events are flagged quiet (not dropped)")
    args = ap.parse_args()

    clip = Path(args.dat).stem
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- source geometry -------------------------------------------
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    scale, pad_x, pad_y = letterbox_params(src_w, src_h)

    print(f"clip        : {clip}")
    print(f"source      : {src_w}x{src_h} @ {fps:.3f} fps, {n_frames} frames")
    print(f"letterbox   : scale {scale:.4f}, pad {pad_x:.1f} / {pad_y:.1f}")
    print(f"window      : {args.window_ms} ms")

    # ---- data ------------------------------------------------------
    boxes_by_frame = load_labels_csv(args.labels)
    events = load_all_events(args.dat)
    t = events["t"]

    print(f"events      : {len(events):,} over "
          f"{t[0] / 1000:.1f}-{t[-1] / 1000:.1f} ms")
    print(f"labelled    : {len(boxes_by_frame)} frames")
    print()

    window_us = int(args.window_ms * 1000)

    tensors = []
    records = []

    n_quiet = 0
    n_empty = 0

    frame_indices = sorted(boxes_by_frame.keys())[::args.stride]

    for frame_idx in frame_indices:
        src_boxes = boxes_by_frame[frame_idx]
        if not src_boxes:
            continue

        frame_t = int(frame_idx / fps * 1_000_000)
        lo = np.searchsorted(t, max(0, frame_t - window_us), side="left")
        hi = np.searchsorted(t, frame_t, side="right")
        window = events[lo:hi]

        # How much event activity actually sits under the labels?
        # A correct box over a hovering drone can contain nothing at all.
        inside = 0
        if len(window):
            ex = window["x"].astype(np.float64)
            ey = window["y"].astype(np.float64)
            mask = np.zeros(len(window), dtype=bool)
            for (bx, by, bw, bh) in src_boxes:
                mask |= (ex >= bx) & (ex < bx + bw) & (ey >= by) & (ey < by + bh)
            inside = int(mask.sum())

        quiet = inside < args.quiet_threshold
        if quiet:
            n_quiet += 1
        if len(window) == 0:
            n_empty += 1

        # ---- tensor -------------------------------------------------
        acc = accumulate_signed(window, src_w, src_h)
        tensor = letterbox_frame(acc, scale, pad_x, pad_y)

        tensors.append(tensor)

        records.append({
            "clip": clip,
            "frame": frame_idx,
            "timestamp_us": frame_t,
            "boxes": [list(transform_box(b, scale, pad_x, pad_y))
                      for b in src_boxes],
            "boxes_src": [list(b) for b in src_boxes],
            "n_events_window": int(len(window)),
            "n_events_in_boxes": inside,
            "quiet": bool(quiet),
        })

    if not tensors:
        raise SystemExit("Nothing written -- no labelled frames with boxes.")

    stack = np.stack(tensors)[..., None]        # (N, 224, 224, 1)

    np.save(out_dir / f"{clip}_tensors.npy", stack)

    meta = {
        "clip": clip,
        "representation": "signed_accumulation_1ch",
        "window_ms": args.window_ms,
        "stride": args.stride,
        "quiet_threshold": args.quiet_threshold,
        "normalization": "none (raw signed counts after INTER_AREA resize)",
        "clipping": "none",
        "source_resolution": [src_w, src_h],
        "input_resolution": [DST, DST],
        "resize": "letterbox, INTER_AREA",
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "fps": fps,
        "n_samples": len(records),
        "n_quiet": n_quiet,
        "n_empty_window": n_empty,
        "dat": str(args.dat),
        "labels": str(args.labels),
        "video": str(args.video),
        "frames": records,
    }

    with open(out_dir / f"{clip}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- report -----------------------------------------------------
    vals = stack[stack != 0]

    print(f"samples written : {len(records)}")
    print(f"tensor stack    : {stack.shape}  ({stack.nbytes / 1e6:.1f} MB)")
    print()
    print(f"quiet frames    : {n_quiet} "
          f"({n_quiet / len(records) * 100:.1f}%) "
          f"- fewer than {args.quiet_threshold} events inside the boxes")
    print(f"empty windows   : {n_empty}")
    print()

    if len(vals):
        print("Non-zero pixel values (informs the normalisation choice):")
        print(f"  min {vals.min():8.2f}   max {vals.max():8.2f}")
        print(f"  p1  {np.percentile(vals, 1):8.2f}   "
              f"p99 {np.percentile(vals, 99):8.2f}")
        print(f"  sparsity: {(stack == 0).mean() * 100:.1f}% of pixels are zero")

    print()
    print(f"Wrote {out_dir / f'{clip}_tensors.npy'}")
    print(f"      {out_dir / f'{clip}_meta.json'}")
    print()
    print("Quiet frames are flagged, not dropped. Decide the policy once")
    print("the rate across clips is known.")


if __name__ == "__main__":
    main()