"""
Convert videos to events, over a selectable slice of the clip list.

The simulator is effectively single-threaded, so one process uses one
core. Splitting the clip list across several processes uses several.

    python generate_events.py --start 0   --end 38
    python generate_events.py --start 38  --end 76
    python generate_events.py --start 76  --end 114

Run those in three separate terminals. Ranges are half-open and indexed
into the sorted video list, so they do not overlap.

Already-converted clips are skipped by default, which also makes the
whole thing resumable after an interruption.
"""

import argparse
import subprocess
import time
from pathlib import Path

import os
# Resolve the project root from this file's location, so these scripts
# work from a clone rather than only from one machine. DRONE_ROOT
# overrides it when the data lives somewhere else.
ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))


# -----------------------------
# PATHS
# -----------------------------

VIDEO_DIR = ROOT / "Data_raw" / "Video_V"
LABELS_DIR = ROOT / "Data_new" / "labels"
OUTPUT_DIR = ROOT / "Data_new" / "events" / "messy_events"

PYTHON = Path(r"C:\tmp\prophesee\py3venv\Scripts\python.exe")

SIMULATOR = ROOT / "src" / "data" / "convert_video_to_events.py"

# -----------------------------
# SIMULATOR PARAMETERS
# -----------------------------

CP = 0.05
CN = 0.05
SHOT_NOISE = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0,
                    help="index of first clip to convert (inclusive)")
    ap.add_argument("--end", type=int, default=0,
                    help="index one past the last clip (0 = to the end)")
    ap.add_argument("--overwrite", action="store_true",
                    help="reconvert clips that already have a .dat")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("V_DRONE_*.mp4"))
    total = len(videos)

    end = args.end if args.end > 0 else total
    videos = videos[args.start:end]

    tag = f"[{args.start}:{end}]"

    print(f"{tag} {len(videos)} of {total} clips")
    print(f"{tag} Cp={CP} Cn={CN} shot_noise={SHOT_NOISE} Hz")
    print(f"{tag} output {OUTPUT_DIR}")
    print()

    converted = 0
    skipped = 0
    existing = 0
    failed = []

    t0 = time.time()

    for i, video in enumerate(videos, 1):
        name = video.stem

        labels = LABELS_DIR / f"{name}_LABELS.csv"
        dat_file = OUTPUT_DIR / f"{name}.dat"

        # No labels means the clip cannot be trained on or verified,
        # so converting it would be several minutes wasted.
        if not labels.exists():
            print(f"{tag} {name}: no labels, skipping")
            skipped += 1
            continue

        if dat_file.exists() and not args.overwrite:
            print(f"{tag} {name}: already converted, skipping")
            existing += 1
            continue

        command = [
            str(PYTHON), str(SIMULATOR), str(video),
            "--Cp", str(CP),
            "--Cn", str(CN),
            "--shot_noise_rate_hz", str(SHOT_NOISE),
            "--quiet",
            "-o", str(dat_file),
        ]

        t_clip = time.time()
        result = subprocess.run(command)
        dt = time.time() - t_clip

        if result.returncode != 0:
            print(f"{tag} {name}: FAILED")
            failed.append(name)
            continue

        converted += 1

        # Rough remaining estimate from this process's own rate.
        done = converted
        rate = (time.time() - t0) / done
        left = (len(videos) - i) * rate

        print(f"{tag} [{i:3d}/{len(videos)}] {name}  "
              f"{dt / 60:.1f} min   ~{left / 3600:.1f} h left")

    print()
    print(f"{tag} converted {converted}, already had {existing}, "
          f"no labels {skipped}, failed {len(failed)}")
    print(f"{tag} elapsed {(time.time() - t0) / 3600:.2f} h")

    for name in failed:
        print(f"{tag}   failed: {name}")


if __name__ == "__main__":
    main()
    