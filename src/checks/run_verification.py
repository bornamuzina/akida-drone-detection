"""
Run verify_event_labels.py across every combination of clip, simulator
preset and accumulation window, then print a summary grid of the median
concentration ratios.

Edit CLIPS / PRESETS / WINDOWS below and run. No arguments.
"""

import re
import subprocess
from pathlib import Path

import os
# Resolve the project root from this file's location, so these scripts
# work from a clone rather than only from one machine. DRONE_ROOT
# overrides it when the data lives somewhere else.
ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))

# -----------------------------
# WHAT TO RUN
# -----------------------------

CLIPS = ["V_DRONE_001", "V_DRONE_045"]

PRESETS = {
    "clean":   ROOT / "Data_new" / "events" / "clean_events",
    "default": ROOT / "Data_new" / "events" / "default_events",
    "messy":   ROOT / "Data_new" / "events" / "messy_events",
}

WINDOWS = [10, 20, 40]

EVERY = 30          # save one image per N frames

# -----------------------------
# PATHS
# -----------------------------

VIDEO_DIR  = ROOT / "Data_raw" / "Video_V"
LABELS_DIR = ROOT / "Data_new" / "labels"
VERIFY_DIR = ROOT / "Data_new" / "verification"

PYTHON = Path(r"C:\tmp\prophesee\py3venv\Scripts\python.exe")
SCRIPT = ROOT / "src" / "checks" / "verify_event_labels.py"

# -----------------------------
# RUN
# -----------------------------

results = {}      # (clip, preset, window) -> median ratio or None
missing = []

for clip in CLIPS:

    video = VIDEO_DIR / f"{clip}.mp4"
    labels = LABELS_DIR / f"{clip}_LABELS.csv"

    if not video.exists():
        print(f"MISSING VIDEO: {video}")
        missing.append(str(video))
        continue

    if not labels.exists():
        print(f"MISSING LABELS: {labels}")
        missing.append(str(labels))
        continue

    for preset, dat_dir in PRESETS.items():

        dat = dat_dir / f"{clip}.dat"

        if not dat.exists():
            print(f"MISSING DAT: {dat}")
            missing.append(str(dat))
            continue

        for window in WINDOWS:

            out = VERIFY_DIR / f"{clip}_{preset}_{window}ms"

            print("=" * 62)
            print(f"{clip}  |  {preset}  |  {window} ms")
            print("=" * 62)

            command = [
                str(PYTHON), str(SCRIPT),
                "--video", str(video),
                "--dat", str(dat),
                "--labels", str(labels),
                "--output", str(out),
                "--window_ms", str(window),
                "--every", str(EVERY),
            ]

            proc = subprocess.run(command, capture_output=True, text=True)

            print(proc.stdout)
            if proc.stderr.strip():
                print(proc.stderr)

            if proc.returncode != 0:
                print(f"FAILED: {clip} {preset} {window}ms")
                results[(clip, preset, window)] = None
                continue

            # Pull the median out of the script's own output rather than
            # recomputing it, so the grid can never disagree with the runs.
            match = re.search(
                r"Median event-density ratio inside boxes:\s*([\d.]+)x",
                proc.stdout)

            results[(clip, preset, window)] = (
                float(match.group(1)) if match else None
            )

# -----------------------------
# SUMMARY
# -----------------------------

print()
print("=" * 62)
print("SUMMARY -- median concentration ratio")
print("=" * 62)
print()

for clip in CLIPS:
    if not any(k[0] == clip for k in results):
        continue

    print(clip)
    header = "  {:<10}".format("preset") + "".join(
        f"{str(w) + ' ms':>12}" for w in WINDOWS)
    print(header)
    print("  " + "-" * (10 + 12 * len(WINDOWS)))

    for preset in PRESETS:
        row = f"  {preset:<10}"
        best = None
        best_w = None

        for window in WINDOWS:
            v = results.get((clip, preset, window))
            if v is None:
                row += f"{'--':>12}"
            else:
                row += f"{str(v) + 'x':>12}"
                if best is None or v > best:
                    best, best_w = v, window

        if best_w is not None:
            row += f"    best: {best_w} ms"

        print(row)

    print()

if missing:
    print("Missing inputs -- these combinations were not run:")
    for m in missing:
        print(f"  {m}")
    print()

print(f"Images written under {VERIFY_DIR}")