"""
Verify that CSV bounding-box labels line up with simulated events.

Reads:
  --video   original RGB mp4 (for resolution / fps / frame count)
  --dat     events produced by the Metavision simulator
  --labels  CSV exported from MATLAB: frame,x,y,w,h  (one row per box)

Writes PNG frames showing the events with the ground-truth boxes drawn on top.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import csv

import cv2
import numpy as np

from metavision_core.event_io import EventsIterator


# ============================================================
# LABELS
# ============================================================

def load_labels_csv(path):
    """
    Read frame,x,y,w,h CSV into {frame_index: [(x, y, w, h), ...]}.

    A frame may legitimately have several boxes (several drones),
    or none at all.
    """
    boxes = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {"frame", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing column(s): {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            frame = int(float(row["frame"]))
            boxes[frame].append((
                float(row["x"]),
                float(row["y"]),
                float(row["w"]),
                float(row["h"]),
            ))

    return boxes


# ============================================================
# EVENTS
# ============================================================

def load_all_events(dat_path):
    """
    Read the whole DAT into one array.

    These clips are ~10 s, so this is a few tens of MB at most --
    far simpler and less error-prone than incremental buffering.
    """
    chunks = []

    iterator = EventsIterator(input_path=str(dat_path), delta_t=100000)

    for chunk in iterator:
        if len(chunk):
            chunks.append(chunk.copy())

    if not chunks:
        raise RuntimeError(f"No events found in {dat_path}")

    return np.concatenate(chunks)


def make_event_image(events, width, height):
    """
    Positive events -> white, negative -> grey, nothing -> black.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if len(events) == 0:
        return img

    x = events["x"].astype(np.int32)
    y = events["y"].astype(np.int32)
    p = events["p"]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]

    pos = p > 0
    img[y[pos], x[pos]] = (255, 255, 255)
    img[y[~pos], x[~pos]] = (110, 110, 110)

    return img


# ============================================================
# DRAWING
# ============================================================

def draw_overlay(img, boxes, frame_idx, timestamp_us, n_events):
    for (bx, by, bw, bh) in boxes:
        x1, y1 = int(round(bx)), int(round(by))
        x2, y2 = int(round(bx + bw)), int(round(by + bh))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    lines = [
        f"Frame {frame_idx}   t = {timestamp_us / 1000:.1f} ms",
        f"events {n_events}   boxes {len(boxes)}",
    ]

    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    return img


# ============================================================
# EVENT DENSITY INSIDE / OUTSIDE THE BOXES
# ============================================================

def box_event_stats(events, boxes, width, height):
    """
    Fraction of this frame's events falling inside any labelled box,
    compared with the fraction of image area the boxes cover.

    If the labels are aligned, events should be far denser inside the
    boxes than chance would predict.
    """
    if len(events) == 0 or not boxes:
        return None

    x = events["x"].astype(np.float64)
    y = events["y"].astype(np.float64)

    inside = np.zeros(len(events), dtype=bool)
    area = 0.0

    for (bx, by, bw, bh) in boxes:
        inside |= (x >= bx) & (x < bx + bw) & (y >= by) & (y < by + bh)
        area += bw * bh

    frac_events = inside.sum() / len(events)
    frac_area = min(area / (width * height), 1.0)

    if frac_area == 0:
        return None

    return frac_events, frac_area, frac_events / frac_area


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--dat", required=True)
    parser.add_argument("--labels", required=True,
                        help="CSV with columns frame,x,y,w,h")
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_ms", type=float, default=20.0)
    parser.add_argument("--every", type=int, default=30,
                        help="save every Nth frame")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- labels -------------------------------------------------
    boxes_by_frame = load_labels_csv(args.labels)
    total_boxes = sum(len(v) for v in boxes_by_frame.values())

    print(f"Labels      : {len(boxes_by_frame)} frames, {total_boxes} boxes")

    # ---- video metadata -----------------------------------------
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video       : {width}x{height} @ {fps:.3f} fps, {frame_count} frames")

    # ---- events --------------------------------------------------
    events = load_all_events(args.dat)

    t = events["t"]
    print(f"Events      : {len(events)} spanning "
          f"{t[0] / 1000:.1f}-{t[-1] / 1000:.1f} ms")

    video_duration_ms = frame_count / fps * 1000
    print(f"Video length: {video_duration_ms:.1f} ms")

    if abs(t[-1] - video_duration_ms * 1000) > 0.1 * video_duration_ms * 1000:
        print("  WARNING: event duration differs from video duration by >10%.")
        print("  Timestamps may not correspond to video time.")

    # Events are time-ordered, so we can binary-search the window edges.
    window_us = int(args.window_ms * 1000)

    # ---- per-frame ----------------------------------------------
    ratios = []
    saved = 0

    for frame_idx in range(frame_count):
        frame_t_us = int(frame_idx / fps * 1_000_000)
        start_us = max(0, frame_t_us - window_us)

        lo = np.searchsorted(t, start_us, side="left")
        hi = np.searchsorted(t, frame_t_us, side="right")
        slice_events = events[lo:hi]

        boxes = boxes_by_frame.get(frame_idx, [])

        stats = box_event_stats(slice_events, boxes, width, height)
        if stats is not None:
            ratios.append(stats[2])

        if frame_idx % args.every == 0:
            img = make_event_image(slice_events, width, height)
            img = draw_overlay(img, boxes, frame_idx, frame_t_us,
                               len(slice_events))

            cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.png"), img)
            saved += 1

            note = ""
            if stats is not None:
                note = f" | inside {stats[0] * 100:5.1f}% of events, ratio {stats[2]:6.1f}x"

            print(f"  frame {frame_idx:4d} | events {len(slice_events):6d} "
                  f"| boxes {len(boxes)}{note}")

    # ---- verdict -------------------------------------------------
    print()
    print("=" * 60)
    print(f"Saved {saved} images to {output_dir}")

    if ratios:
        median = float(np.median(ratios))
        print(f"Median event-density ratio inside boxes: {median:.1f}x")
        print()
        if median > 5:
            print("Strongly concentrated inside the boxes -- labels look aligned.")
        elif median > 2:
            print("Somewhat concentrated. Probably aligned, but check the images.")
        else:
            print("No real concentration. Either the labels are misaligned,")
            print("or noise dominates. Inspect the images before trusting this.")
    else:
        print("No frames had both events and boxes -- nothing to measure.")

    print("=" * 60)
    print()
    print("Look at the images: the green rectangles should sit on the")
    print("bright event clusters. The ratio is a summary, not a substitute.")


if __name__ == "__main__":
    main()
