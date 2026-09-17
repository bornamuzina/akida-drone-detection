"""
Build RGB training data from the source videos.

This is the baseline the event-based work will be compared against, so
the geometry has to match exactly: same letterbox, same 224x224, same
anchors, same splits. Only the pixels differ -- ordinary video frames
instead of accumulated events.

Two outputs, because the two models want different formats:

  --format npy    (N, 224, 224, 3) stacks plus a metadata json, the same
                  shape the existing YOLOv2 pipeline already reads

  --format yolo   individual jpg files with one txt per image, which is
                  what Ultralytics expects

RGB needs no clipping or rescaling: frames are already 0-255, which is
what the model's Rescaling layer wants.

Usage:
    python build_rgb_tensors.py --format npy
    python build_rgb_tensors.py --format yolo
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import shutil

import cv2
import numpy as np


ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))

VIDEO_DIR = ROOT / "Data_raw" / "Video_V"
LABELS_DIR = ROOT / "Data_new" / "labels"
SPLITS_FILE = ROOT / "Data_new" / "splits.json"

DST = 256


# ============================================================
# GEOMETRY  (identical to build_tensors.py)
# ============================================================

def letterbox_params(src_w, src_h, dst=DST):
    scale = min(dst / src_w, dst / src_h)
    return scale, (dst - src_w * scale) / 2.0, (dst - src_h * scale) / 2.0


def letterbox_image(img, scale, pad_x, pad_y, dst=DST):
    """
    Resize preserving aspect ratio, then pad into a square.

    INTER_AREA for the same reason as the event pipeline: it averages
    the source pixels covering each output pixel, which is the right
    choice when shrinking by 0.35.
    """
    src_h, src_w = img.shape[:2]
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((dst, dst, 3), dtype=np.uint8)
    y0, x0 = int(round(pad_y)), int(round(pad_x))
    out[y0:y0 + new_h, x0:x0 + new_w] = resized

    return out


def transform_box(box, scale, pad_x, pad_y):
    x, y, w, h = box
    return (x * scale + pad_x, y * scale + pad_y, w * scale, h * scale)


# ============================================================
# LABELS
# ============================================================

def load_labels_csv(path):
    boxes = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            boxes[int(float(row["frame"]))].append((
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            ))
    return boxes


def clip_name(csv_path):
    stem = csv_path.stem
    return stem[:-len("_LABELS")] if stem.endswith("_LABELS") else stem


# ============================================================
# NPY FORMAT  (for the existing YOLOv2 pipeline)
# ============================================================

def build_npy(clip, video, boxes_by_frame, out_dir):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale, pad_x, pad_y = letterbox_params(src_w, src_h)

    wanted = set(boxes_by_frame)

    frames = []
    records = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in wanted and boxes_by_frame[idx]:
            frames.append(letterbox_image(frame, scale, pad_x, pad_y))
            records.append({
                "clip": clip,
                "frame": idx,
                "timestamp_us": int(idx / fps * 1_000_000),
                "boxes": [list(transform_box(b, scale, pad_x, pad_y))
                          for b in boxes_by_frame[idx]],
                "boxes_src": [list(b) for b in boxes_by_frame[idx]],
                # No quiet concept for RGB -- a frame always has pixels,
                # unlike an event window which can be empty. Kept in the
                # schema so dataset.py needs no changes.
                "n_events_window": 0,
                "n_events_in_boxes": 0,
                "quiet": False,
            })

        idx += 1

    cap.release()

    if not frames:
        return None

    stack = np.stack(frames)          # (N, 224, 224, 3) uint8

    np.save(out_dir / f"{clip}_tensors.npy", stack)

    meta = {
        "clip": clip,
        "representation": "rgb_3ch",
        "normalization": "none (raw 0-255)",
        "clipping": "none",
        "source_resolution": [src_w, src_h],
        "input_resolution": [DST, DST],
        "resize": "letterbox, INTER_AREA",
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "fps": fps,
        "n_samples": len(records),
        "n_quiet": 0,
        "n_empty_window": 0,
        "video": str(video),
        "frames": records,
    }

    with open(out_dir / f"{clip}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return len(records)


# ============================================================
# YOLO FORMAT  (for Ultralytics)
# ============================================================

def build_yolo(clip, video, boxes_by_frame, split, out_root, stride=1):
    """
    One jpg per frame, one txt beside it.

    Ultralytics wants normalised centre-form coordinates:
        class cx cy w h      all divided by image size
    """
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale, pad_x, pad_y = letterbox_params(src_w, src_h)

    wanted = set(boxes_by_frame)

    written = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in wanted and boxes_by_frame[idx] and idx % stride == 0:
            img = letterbox_image(frame, scale, pad_x, pad_y)

            name = f"{clip}_{idx:06d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            lines = []
            for box in boxes_by_frame[idx]:
                x, y, w, h = transform_box(box, scale, pad_x, pad_y)
                cx = (x + w / 2) / DST
                cy = (y + h / 2) / DST
                nw = w / DST
                nh = h / DST
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            written += 1

        idx += 1

    cap.release()
    return written


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["npy", "yolo"], default="npy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="yolo format: keep every Nth frame")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    split_of = {}
    for name in ("train", "validation", "test"):
        for clip in splits[name]:
            # Ultralytics uses "val", not "validation".
            split_of[clip] = "val" if name == "validation" else name

    csvs = sorted(LABELS_DIR.glob("*.csv"))
    if args.limit:
        csvs = csvs[:args.limit]

    if args.format == "npy":
        out_dir = Path(args.out) if args.out else ROOT / "Data_new" / "tensors_rgb" / f"tensors_rgb_{DST}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.out) if args.out else ROOT / "Data_new" / "tensors_rgb" / f"yolo26_rgb_{DST}_ultralytics"
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"format : {args.format}")
    print(f"output : {out_dir}")
    print(f"clips  : {len(csvs)}")
    print()

    total = 0
    done = 0
    missing = []

    for i, csv_path in enumerate(csvs, 1):
        clip = clip_name(csv_path)
        video = VIDEO_DIR / f"{clip}.mp4"

        if not video.exists():
            missing.append(clip)
            continue

        if clip not in split_of:
            # Not in any split -- skip rather than guess.
            missing.append(clip)
            continue

        boxes = load_labels_csv(csv_path)

        if args.format == "npy":
            n = build_npy(clip, video, boxes, out_dir)
        else:
            n = build_yolo(clip, video, boxes, split_of[clip], out_dir,
                           stride=args.stride)

        if not n:
            missing.append(clip)
            continue

        total += n
        done += 1
        print(f"[{i:3d}/{len(csvs)}] {clip}  {n} frames  ({split_of[clip]})")

    print()
    print(f"clips  : {done}")
    print(f"frames : {total:,}")
    if missing:
        print(f"skipped: {len(missing)}")

    # ---- extras -------------------------------------------------------
    if args.format == "npy":
        shutil.copy2(SPLITS_FILE, out_dir / "splits.json")
        print()
        print("copied splits.json")
        print()
        print("config.py needs CHANNELS = 3 and TENSOR_DIR pointed here.")

    else:
        # Ultralytics reads a small yaml describing the dataset.
        yaml = (
            f"path: {out_dir.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "\n"
            "names:\n"
            "  0: drone\n"
        )
        (out_dir / "data.yaml").write_text(yaml)

        print()
        print(f"wrote {out_dir / 'data.yaml'}")
        print()
        print("Train with:")
        print(f"  yolo detect train data={out_dir / 'data.yaml'} "
              "model=yolo11n.pt imgsz=224 epochs=50")


if __name__ == "__main__":
    main()