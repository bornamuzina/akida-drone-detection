"""
Build tensors for every clip that has labels.

Wraps build_tensors.py, runs it once per clip, then reports statistics
across the whole dataset -- particularly the quiet-frame rate, which is
the open label-policy question.

Usage:
    python build_all_tensors.py
    python build_all_tensors.py --window_ms 10 --output_suffix _10ms
"""

import os
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))

LABELS_DIR = ROOT / "Data_new" / "labels"
VIDEO_DIR = ROOT / "Data_raw" / "Video_V"
SCRIPT = ROOT / "scripts" / "build_tensors.py"


def clip_name(csv_path):
    """V_DRONE_106_LABELS.csv -> V_DRONE_106"""
    stem = csv_path.stem
    return stem[:-len("_LABELS")] if stem.endswith("_LABELS") else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_dir", default=str(ROOT / "Data_new" / "events" / "messy_events"))
    ap.add_argument("--output", default=str(ROOT / "Data_new" / "tensors_events" / "tensors_new"))
    ap.add_argument("--window_ms", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N clips (0 = all); useful for a dry run")
    args = ap.parse_args()

    events_dir = Path(args.events_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(LABELS_DIR.glob("*.csv"))
    if args.limit:
        csvs = csvs[:args.limit]

    print(f"label files : {len(csvs)}")
    print(f"events      : {events_dir}")
    print(f"output      : {out_dir}")
    print(f"window      : {args.window_ms} ms")
    print("=" * 64)

    done = []
    skipped = []
    failed = []

    t_start = time.time()

    for i, csv_path in enumerate(csvs, 1):
        clip = clip_name(csv_path)

        dat = events_dir / f"{clip}.dat"
        video = VIDEO_DIR / f"{clip}.mp4"

        if not dat.exists():
            skipped.append((clip, "no .dat"))
            continue
        if not video.exists():
            skipped.append((clip, "no .mp4"))
            continue

        cmd = [
            sys.executable, str(SCRIPT),
            "--dat", str(dat),
            "--labels", str(csv_path),
            "--video", str(video),
            "--output", str(out_dir),
            "--window_ms", str(args.window_ms),
            "--stride", str(args.stride),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            failed.append((clip, proc.stderr.strip().splitlines()[-1:]))
            print(f"[{i:3d}/{len(csvs)}] {clip}  FAILED")
            continue

        done.append(clip)
        print(f"[{i:3d}/{len(csvs)}] {clip}  ok")

    elapsed = time.time() - t_start

    # ---- aggregate the metadata -------------------------------------
    print()
    print("=" * 64)
    print(f"built   : {len(done)}")
    print(f"skipped : {len(skipped)}")
    print(f"failed  : {len(failed)}")
    print(f"time    : {elapsed / 60:.1f} min")

    for clip, why in skipped[:10]:
        print(f"  skipped {clip}: {why}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")

    for clip, why in failed[:10]:
        print(f"  failed {clip}: {why}")

    if not done:
        return

    total_samples = 0
    total_quiet = 0
    total_boxes = 0
    total_bytes = 0
    per_clip = []

    for clip in done:
        meta_path = out_dir / f"{clip}_meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        n = meta["n_samples"]
        q = meta["n_quiet"]
        b = sum(len(fr["boxes"]) for fr in meta["frames"])

        total_samples += n
        total_quiet += q
        total_boxes += b

        npy = out_dir / f"{clip}_tensors.npy"
        if npy.exists():
            total_bytes += npy.stat().st_size

        per_clip.append((clip, n, q, q / n * 100 if n else 0))

    print()
    print("DATASET")
    print(f"  clips        : {len(per_clip)}")
    print(f"  samples      : {total_samples:,}")
    print(f"  boxes        : {total_boxes:,}")
    print(f"  on disk      : {total_bytes / 1e9:.2f} GB")
    print()
    print(f"  quiet frames : {total_quiet:,} "
          f"({total_quiet / total_samples * 100:.1f}% of samples)")

    # The clips where the quiet problem actually lives.
    per_clip.sort(key=lambda r: -r[3])
    print()
    print("  worst clips by quiet rate:")
    for clip, n, q, pct in per_clip[:10]:
        if q == 0:
            break
        print(f"    {clip:20s} {q:4d}/{n:4d}  {pct:5.1f}%")

    n_bad = sum(1 for _, _, _, pct in per_clip if pct > 20)
    print()
    print(f"  clips with >20% quiet frames: {n_bad}")

    # ---- a manifest, so the split step has something to read ---------
    manifest = {
        "events_dir": str(events_dir),
        "window_ms": args.window_ms,
        "stride": args.stride,
        "clips": [
            {"clip": c, "samples": n, "quiet": q}
            for c, n, q, _ in per_clip
        ],
        "total_samples": total_samples,
        "total_boxes": total_boxes,
        "total_quiet": total_quiet,
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
    