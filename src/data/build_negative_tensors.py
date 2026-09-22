"""
Convert the non-drone clips into tensors, as negatives.

Why this is a separate script
-----------------------------
build_rgb_tensors.py is driven by the label CSVs: it walks
Data_new/labels/*.csv and keeps only frames that have a box. Negatives
have no box by definition, so that loop would produce nothing. They are
also not in the label export at all -- and they do not need to be, since
their own class (airplane, bird, helicopter) is deliberately ignored.

So this walks the clip lists instead, reads the video directly, and
writes every frame with an empty box list.

The geometry is imported from build_rgb_tensors rather than copied.
Negatives that were letterboxed even slightly differently from the
positives would be a bug nobody would ever find by looking.

Stride
------
Consecutive frames of the same empty sky are near-duplicates: they cost
disk and training time and teach almost nothing beyond the first. So
only every Nth frame is kept. At the default of 3 the 171 clips give
roughly 17,000 negatives against 35,496 positives, which is about 3 GB
and a ratio the training can actually use.

Usage
-----
    python build_negative_tensors.py --dry_run
    python build_negative_tensors.py
    python build_negative_tensors.py --out <dir> --stride 3
"""

from pathlib import Path
import argparse
import json

import cv2
import numpy as np

from build_rgb_tensors import (
    ROOT, VIDEO_DIR, SPLITS_FILE, DST,
    letterbox_params, letterbox_image,
)


NEG_KEYS = ("train_neg", "validation_neg", "test_neg")


def build_negative(clip, video, out_dir, stride, compress=True):
    """
    Write one clip's frames with no boxes.

    Returns the number of frames written, or None if the video could
    not be opened or held nothing.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale, pad_x, pad_y = letterbox_params(src_w, src_h)

    frames = []
    records = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % stride == 0:
            frames.append(letterbox_image(frame, scale, pad_x, pad_y))
            records.append({
                "clip": clip,
                "frame": idx,
                "timestamp_us": int(idx / fps * 1_000_000),
                # The whole point: present, and empty.
                "boxes": [],
                "boxes_src": [],
                "n_events_window": 0,
                "n_events_in_boxes": 0,
                "quiet": False,
                "negative": True,
            })

        idx += 1

    cap.release()

    if not frames:
        return None

    stack = np.stack(frames)

    if compress:
        # The existing clips in the packed folder are compressed .npz
        # under the key "a", which is what dataset.load_clip reads.
        # Matching them keeps one format per folder and costs about a
        # fifth of the disk -- sky compresses well.
        np.savez_compressed(out_dir / f"{clip}_tensors.npz", a=stack)
    else:
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
        "negative": True,
        "stride": stride,
        "note": (
            "Negative clip. Its own class (airplane / bird / helicopter) "
            "is ignored entirely and every frame carries an empty box "
            "list, so the model stays single-class."
        ),
        "frames": records,
    }

    with open(out_dir / f"{clip}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=3,
                    help="keep every Nth frame (default 3)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compress", action="store_true",
                    help="write .npz directly instead of plain .npy; "
                         "normally left off and pack_for_upload.py does it")
    ap.add_argument("--dry_run", action="store_true",
                    help="count frames and estimate size, write nothing")
    args = ap.parse_args()

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    missing_keys = [k for k in NEG_KEYS if k not in splits]
    if missing_keys:
        raise SystemExit(
            f"{splits_missing(missing_keys)}\n"
            "Run make_negative_splits.py first."
        )

    clips = []
    for key in NEG_KEYS:
        for clip in splits[key]:
            clips.append((clip, key))

    if args.limit:
        clips = clips[:args.limit]

    out_dir = (Path(args.out) if args.out else
               ROOT / "Data_new" / "tensors_rgb" / f"tensors_rgb_{DST}_neg")

    compress = args.compress

    print()
    print(f"output : {out_dir}")
    print(f"stride : 1 frame in {args.stride}")
    print(f"format : {'npz (compressed)' if compress else 'npy'}")
    print(f"clips  : {len(clips)}")
    print()

    # ---- dry run: count without decoding every frame ----------------
    if args.dry_run:
        total = 0
        for clip, _ in clips:
            video = VIDEO_DIR / f"{clip}.mp4"
            if not video.exists():
                print(f"  MISSING {clip}")
                continue

            cap = cv2.VideoCapture(str(video))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            total += (n + args.stride - 1) // args.stride

        raw = total * DST * DST * 3 / 1e9
        gb = raw / 5 if compress else raw

        print(f"frames it would write : {total:,}")
        print(f"disk                  : about {gb:.1f} GB"
              f"{'  (compressed; ~' + format(raw, '.1f') + ' GB raw)' if compress else ''}")
        print()
        print("Nothing written. Drop --dry_run to build.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    done = 0
    missing = []

    for i, (clip, key) in enumerate(clips, 1):
        video = VIDEO_DIR / f"{clip}.mp4"

        if not video.exists():
            missing.append(clip)
            continue

        n = build_negative(clip, video, out_dir, args.stride,
                           compress=compress)

        if not n:
            missing.append(clip)
            continue

        total += n
        done += 1
        print(f"[{i:3d}/{len(clips)}] {clip}  {n} frames  ({key})")

    print()
    print(f"clips  : {done}")
    print(f"frames : {total:,}")

    if missing:
        print(f"skipped: {len(missing)}")
        for c in missing[:10]:
            print(f"         {c}")
        if len(missing) > 10:
            print(f"         ... and {len(missing) - 10} more")

    print()
    print("Next:")
    print("  1. pack them the same way the drone clips were packed:")
    print("       python pack_for_upload.py --tensors <this dir> --out <packed>")
    print("  2. move the .npz and _meta.json into the folder config.py")
    print("     points at, so load_clip finds them by name")
    print()
    print("dataset.py still drops frames with no boxes, so nothing reaches")
    print("training until that is changed.")


def splits_missing(keys):
    return f"splits.json has no {', '.join(keys)}."


if __name__ == "__main__":
    main()