"""
Split the clips into train / validation / test.

Split is BY CLIP, never by frame. Consecutive frames in a clip are nearly
identical, so a frame-level split would put near-copies of training data
into the test set and produce a flattering, meaningless score.

Clips are stratified by median target size, because target size is the
dominant difficulty factor in this dataset (90% of boxes are under 16 px
in model space). A random split could easily hand the test set all the
easy close-range clips, or all the hard distant ones.

Sizes are read from the label CSVs, not from the tensors, so the split
does not depend on which preset has been generated.

Usage:
    python make_splits.py
    python make_splits.py --n_val 12 --n_test 12 --seed 0
"""

import os
from pathlib import Path
import argparse
import csv
import json

import numpy as np


ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))
LABELS_DIR = ROOT / "Data_new" / "labels"

# Letterbox scale, so sizes are reported in the space the model sees.
SCALE = 224 / 640

# Clips already measured and written up in detail. Held out for test so
# the existing analysis describes data the model never trained on.
PIN_TEST = ["V_DRONE_001", "V_DRONE_045", "V_DRONE_106"]


def clip_name(csv_path):
    stem = csv_path.stem
    return stem[:-len("_LABELS")] if stem.endswith("_LABELS") else stem


def clip_stats(csv_path):
    """Median box size (in 224-space px) and number of boxes."""
    sizes = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            w = float(row["w"]) * SCALE
            h = float(row["h"]) * SCALE
            if w > 0 and h > 0:
                sizes.append(np.sqrt(w * h))

    if not sizes:
        return None

    return {
        "n_boxes": len(sizes),
        "median_size": float(np.median(sizes)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--n_test", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(ROOT / "Data_new" / "splits.json"))
    args = ap.parse_args()

    # ---- gather -----------------------------------------------------
    clips = {}
    for csv_path in sorted(LABELS_DIR.glob("*.csv")):
        st = clip_stats(csv_path)
        if st:
            clips[clip_name(csv_path)] = st

    print(f"clips with boxes : {len(clips)}")

    names = sorted(clips)
    sizes = np.array([clips[n]["median_size"] for n in names])

    print(f"median target size across clips: "
          f"{np.median(sizes):.1f} px "
          f"(min {sizes.min():.1f}, max {sizes.max():.1f})")

    # ---- stratify ---------------------------------------------------
    # Sort by size and walk through in blocks, drawing val/test picks
    # from each block. This keeps the difficulty range represented in
    # all three sets rather than trusting a random draw.
    order = np.argsort(sizes)
    ordered = [names[i] for i in order]

    pinned = [c for c in PIN_TEST if c in clips]
    missing = [c for c in PIN_TEST if c not in clips]
    if missing:
        print(f"WARNING: pinned clips not found: {missing}")

    pool = [c for c in ordered if c not in pinned]

    n_test_draw = max(0, args.n_test - len(pinned))
    n_draw = args.n_val + n_test_draw

    rng = np.random.default_rng(args.seed)

    # One pick per block, so picks are spread across the size range.
    n_blocks = n_draw
    block_size = len(pool) / n_blocks

    picks = []
    for i in range(n_blocks):
        lo = int(round(i * block_size))
        hi = int(round((i + 1) * block_size))
        hi = max(hi, lo + 1)
        block = pool[lo:min(hi, len(pool))]
        if not block:
            continue
        picks.append(block[rng.integers(len(block))])

    picks = list(dict.fromkeys(picks))          # dedupe, keep order

    # Top up if blocks collided.
    remaining = [c for c in pool if c not in picks]
    while len(picks) < n_draw and remaining:
        picks.append(remaining.pop(rng.integers(len(remaining))))

    rng.shuffle(picks)

    val = sorted(picks[:args.n_val])
    test = sorted(pinned + picks[args.n_val:args.n_val + n_test_draw])
    train = sorted(c for c in names if c not in val and c not in test)

    # ---- report -----------------------------------------------------
    def describe(label, group):
        s = np.array([clips[c]["median_size"] for c in group])
        boxes = sum(clips[c]["n_boxes"] for c in group)
        print(f"  {label:11s} {len(group):3d} clips  "
              f"{boxes:6d} boxes  "
              f"median size {np.median(s):5.1f} px  "
              f"range {s.min():4.1f}-{s.max():4.1f}")

    print()
    print("SPLIT")
    describe("train", train)
    describe("validation", val)
    describe("test", test)

    print()
    print("validation:", ", ".join(val))
    print()
    print("test      :", ", ".join(test))

    # A quick sanity check that stratification did its job. If the median
    # target size differs wildly between splits, the sets are not
    # comparable and the numbers will mislead.
    med_tr = np.median([clips[c]["median_size"] for c in train])
    med_va = np.median([clips[c]["median_size"] for c in val])
    med_te = np.median([clips[c]["median_size"] for c in test])

    spread = max(med_tr, med_va, med_te) / min(med_tr, med_va, med_te)

    print()
    print(f"median-size spread across splits: {spread:.2f}x")
    if spread > 1.3:
        print("  Splits differ noticeably in difficulty. Consider another seed.")
    else:
        print("  Splits are comparable in difficulty.")

    # ---- write ------------------------------------------------------
    out = {
        "split_unit": "clip",
        "rationale": "consecutive frames are near-duplicates; a frame-level "
                     "split would leak training data into the test set",
        "stratified_by": "median target size in 224x224 space",
        "seed": args.seed,
        "pinned_to_test": pinned,
        "train": train,
        "validation": val,
        "test": test,
        "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "per_clip": clips,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"Wrote {args.output}")
    print("These clip lists are fixed from here on. Changing them after")
    print("looking at test results would invalidate the test score.")


if __name__ == "__main__":
    main()