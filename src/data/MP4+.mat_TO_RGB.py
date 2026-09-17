from pathlib import Path
import cv2
from mcos_decoder import load_groundtruth

import os
# Resolve the project root from this file's location, so these scripts
# work from a clone rather than only from one machine. DRONE_ROOT
# overrides it when the data lives somewhere else.
ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))

# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = ROOT / "Data_raw" / "Video_V"

OUTPUT_DIR = ROOT / "Data_new" / "annotated_rgb"

# Create output folder
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# FIND ALL RGB VIDEOS
# ============================================================

videos = sorted(INPUT_DIR.glob("*.mp4"))

print("=" * 70)
print(f"Found {len(videos)} RGB videos")
print("=" * 70)

# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for video_number, video_path in enumerate(videos, start=1):

    # --------------------------------------------------------
    # Find matching MAT label file
    # --------------------------------------------------------

    label_path = video_path.with_name(
        video_path.stem + "_LABELS.mat"
    )

    if not label_path.exists():
        print()
        print(f"WARNING: No label file for {video_path.name}")
        print(f"Expected: {label_path.name}")
        continue

    # --------------------------------------------------------
    # Determine class
    # --------------------------------------------------------

    name = video_path.stem.upper()

    if "AIRPLANE" in name:
        class_name = "AIRPLANE"

    elif "BIRD" in name:
        class_name = "BIRD"

    elif "DRONE" in name:
        class_name = "DRONE"

    elif "HELICOPTER" in name:
        class_name = "HELICOPTER"

    else:
        class_name = "UNKNOWN"

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / (
        f"{video_path.stem}_annotated.mp4"
    )

    print()
    print("=" * 70)
    print(f"[{video_number}/{len(videos)}] {video_path.name}")
    print(f"Class:  {class_name}")
    print(f"Labels: {label_path.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    try:
        bboxes = load_groundtruth(str(label_path))

    except Exception as e:
        print(f"ERROR loading labels: {e}")
        continue

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"Resolution: {width} x {height}")
    print(f"FPS:        {fps:.2f}")
    print(f"Video frames: {frame_count}")
    print(f"Label frames: {len(bboxes)}")

    # --------------------------------------------------------
    # Create output video
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video")
        cap.release()
        continue

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Get annotation for current frame
        # ----------------------------------------------------

        if frame_idx < len(bboxes):
            bbox = bboxes[frame_idx]
        else:
            bbox = None

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        if bbox is not None:

            x, y, w, h = bbox

            x1 = int(x)
            y1 = int(y)

            x2 = int(x + w)
            y2 = int(y + h)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Class label
            cv2.putText(
                frame,
                class_name,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # Frame number
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_idx + 1}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(frame)

        frame_idx += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_idx % 100 == 0:

            percent = (
                frame_idx / frame_count * 100
                if frame_count > 0
                else 0
            )

            print(
                f"\rProgress: "
                f"{frame_idx}/{frame_count} "
                f"({percent:.1f}%)",
                end=""
            )

    # --------------------------------------------------------
    # Close video
    # --------------------------------------------------------

    cap.release()
    writer.release()

    print()
    print(f"Saved -> {output_path}")

# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("ALL RGB VIDEOS FINISHED")
print("=" * 70)
print(f"Output folder:")
print(OUTPUT_DIR)