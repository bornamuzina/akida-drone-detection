import subprocess
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
OUTPUT_DIR = ROOT / "Data_new" / "events" / "messy_events"   #   <--------------

PYTHON = Path(r"C:\tmp\prophesee\py3venv\Scripts\python.exe")

# Prophesee's sample script -- do NOT use for conversion.
# It only writes what its 50,000-event display buffer holds,
# so sparse clips lose most of their events.
# SIMULATOR = Path(
#     r"C:\dev\openeb\sdk\modules\core_ml\python"
#     r"\samples\viz_video_to_event_simulator"
#     r"\viz_video_to_event_simulator.py"
# )

# Our converter -- writes every event.
SIMULATOR = ROOT / "src" / "data" / "convert_video_to_events.py"

# -----------------------------
# SIMULATOR PARAMETERS
# -----------------------------

CP = 0.05                                                         #   <--------------
CN = 0.05
SHOT_NOISE = 15

# -----------------------------
# RUN
# -----------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

videos = sorted(VIDEO_DIR.glob("V_DRONE_*.mp4"))

print(f"Found {len(videos)} videos.")
print(f"Cp={CP}  Cn={CN}  shot_noise={SHOT_NOISE} Hz")
print(f"Output: {OUTPUT_DIR}")
print()

converted = 0
skipped = 0
failed = []

for video in videos:

    name = video.stem

    labels = LABELS_DIR / f"{name}_LABELS.csv"
    dat_file = OUTPUT_DIR / f"{name}.dat"

    print("=" * 60)
    print(f"Processing: {name}")
    print(f"Video:  {video}")
    print(f"Events: {dat_file}")

    # A clip with no label CSV cannot be verified or used for
    # training, so there is no point spending minutes converting it.
    if not labels.exists():
        print("WARNING: no label CSV found. Skipping.")
        skipped += 1
        continue

    command = [
        str(PYTHON),
        str(SIMULATOR),
        str(video),

        "--Cp", str(CP),
        "--Cn", str(CN),
        "--shot_noise_rate_hz", str(SHOT_NOISE),

        "-o", str(dat_file)
    ]

    print()
    print("Running simulator...")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"ERROR: simulator failed for {name}")
        failed.append(name)
        continue

    converted += 1
    print(f"Finished: {name}")
    print()

print("=" * 60)
print("ALL DONE")
print(f"Converted: {converted}")
print(f"Skipped (no labels): {skipped}")

if failed:
    print(f"Failed: {len(failed)}")
    for name in failed:
        print(f"  {name}")

print(f"Output folder: {OUTPUT_DIR}")