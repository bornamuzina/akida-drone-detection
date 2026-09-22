"""
Add the non-drone clips to splits.json as negatives.

Why this exists
---------------
Every frame in the current dataset contains a drone, so the false-alarm
rate has never been measured. The source dataset also holds 171 clips of
airplanes, birds and helicopters that were never used. Treated as
negatives -- frames with no box at all, the other classes ignored -- they
give the model something to learn "nothing here" from, and give the
evaluation a number it currently cannot produce.

What it does
------------
Reads Data_raw/Video_V, takes every clip whose name is not V_DRONE_*,
and splits it 79 / 10.5 / 10.5 the same way the drone clips were split.
The split is stratified per class, so airplanes, birds and helicopters
each appear in all three parts. If every helicopter landed in train, the
test set could not say whether helicopters are handled.

The existing train / validation / test keys are NOT touched. Every
result so far depends on that division, and regenerating it with a
different seed would silently invalidate all of them. The negatives go
in under their own keys:

    train_neg   validation_neg   test_neg

so the old evaluation still runs unchanged, and a negative-inclusive
one is a deliberate choice rather than an accident.

Usage
-----
    python make_negative_splits.py            # show what it would do
    python make_negative_splits.py --write    # write it
"""

from pathlib import Path
from collections import defaultdict
import argparse
import json

# This script lives one level down from the model code it imports, so
# put the parent directory on the path before importing any of it.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


SEED = 0

# The drone clips went 90 / 12 / 12 out of 114.
TRAIN_FRAC = 90 / 114
VAL_FRAC = 12 / 114


def find_clips(video_dir):
    """Clip stems from the .mp4 files, grouped by class."""
    by_class = defaultdict(list)

    for path in sorted(video_dir.glob("*.mp4")):
        parts = path.stem.split("_")

        if len(parts) < 2:
            continue

        by_class[parts[1].upper()].append(path.stem)

    return by_class


def split_list(items, train_frac, val_frac, rng):
    """Shuffle once, then cut. Shuffling keeps clip order from leaking
    location or weather into one split."""
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    # Whatever rounding leaves over goes to test, which is the part
    # that most needs not to be empty.
    return (items[:n_train],
            items[n_train:n_train + n_val],
            items[n_train + n_val:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually modify splits.json")
    ap.add_argument("--video_dir", default=None)
    args = ap.parse_args()

    import random

    splits_file = Path(config.SPLITS_FILE)
    video_dir = (Path(args.video_dir) if args.video_dir
                 else Path(config.ROOT) / "Data_raw" / "Video_V")

    print()
    print(f"splits : {splits_file}")
    print(f"videos : {video_dir}")
    print()

    if not video_dir.exists():
        raise SystemExit(f"No such directory: {video_dir}")

    with open(splits_file) as f:
        splits = json.load(f)

    by_class = find_clips(video_dir)

    if not by_class:
        raise SystemExit("No .mp4 files found.")

    print("CLIPS FOUND")
    for cls in sorted(by_class):
        print(f"  {cls:<12}{len(by_class[cls]):>5}")
    print()

    # ---- sanity: the drone clips should match what splits.json holds
    n_drone_existing = sum(len(splits.get(k, []))
                           for k in ("train", "validation", "test"))
    n_drone_found = len(by_class.get("DRONE", []))

    print(f"drone clips in splits.json : {n_drone_existing}")
    print(f"drone clips on disk        : {n_drone_found}")

    if n_drone_existing != n_drone_found:
        print()
        print("  These disagree. Not necessarily wrong -- a clip may have")
        print("  been dropped deliberately -- but worth knowing before")
        print("  adding anything.")

    print()

    # ---- the negatives ----------------------------------------------
    rng = random.Random(SEED)

    out = {"train_neg": [], "validation_neg": [], "test_neg": []}
    per_class = {}

    print("NEGATIVES")
    print(f"  {'class':<12}{'total':>7}{'train':>8}{'val':>6}{'test':>6}")

    for cls in sorted(k for k in by_class if k != "DRONE"):
        tr, va, te = split_list(by_class[cls], TRAIN_FRAC, VAL_FRAC, rng)

        out["train_neg"] += tr
        out["validation_neg"] += va
        out["test_neg"] += te

        per_class[cls] = {"total": len(by_class[cls]), "train": len(tr),
                          "validation": len(va), "test": len(te)}

        print(f"  {cls:<12}{len(by_class[cls]):>7}{len(tr):>8}"
              f"{len(va):>6}{len(te):>6}")

    print(f"  {'TOTAL':<12}{sum(len(v) for v in out.values()):>7}"
          f"{len(out['train_neg']):>8}{len(out['validation_neg']):>6}"
          f"{len(out['test_neg']):>6}")

    for k in out:
        out[k].sort()

    # ---- ratio warning ----------------------------------------------
    print()
    n_pos = len(splits.get("train", []))
    n_neg = len(out["train_neg"])

    print(f"  train: {n_pos} drone clips, {n_neg} negative clips "
          f"({n_neg / max(n_pos, 1):.2f}x)")
    print()
    print("  With more negatives than positives, the cheapest way for the")
    print("  model to cut its loss is to stop detecting. Cap the share of")
    print("  negatives per batch in train.py rather than feeding them all.")

    # ---- collisions --------------------------------------------------
    existing = set()
    for k in ("train", "validation", "test"):
        existing |= set(splits.get(k, []))

    clash = existing & set(sum(out.values(), []))
    if clash:
        raise SystemExit(f"{len(clash)} clips are already in a positive "
                         f"split, e.g. {sorted(clash)[:3]} -- aborting.")

    already = [k for k in out if k in splits]

    print()

    if not args.write:
        print("Dry run. Nothing written. Re-run with --write.")
        if already:
            print(f"NOTE: {already} already exist and would be replaced.")
        return

    if already:
        backup = splits_file.with_suffix(".json.bak")
        backup.write_text(json.dumps(splits, indent=2))
        print(f"{already} already present -- previous file copied to")
        print(f"  {backup}")

    splits.update(out)

    # The existing metadata keys (counts, per_clip, seed, rationale and
    # so on) describe the drone split and stay exactly as they are --
    # they are still true of train / validation / test. This block
    # describes the negatives separately rather than editing them, so
    # nothing that was recorded before is lost or quietly changed.
    splits["negatives"] = {
        "keys": ["train_neg", "validation_neg", "test_neg"],
        "source": "Data_raw/Video_V, every clip not named V_DRONE_*",
        "classes": sorted(k for k in by_class if k != "DRONE"),
        "split_unit": "clip",
        "stratified_by": "class",
        "seed": SEED,
        "fractions": {"train": round(TRAIN_FRAC, 6),
                      "validation": round(VAL_FRAC, 6),
                      "test": round(1 - TRAIN_FRAC - VAL_FRAC, 6)},
        "counts": per_class,
        "labels": (
            "Used as negatives: the other classes are ignored entirely "
            "and every frame carries an empty box list. The model stays "
            "single-class (drone)."
        ),
        "rationale": (
            "No frame in the drone split is drone-free, so the "
            "false-alarm rate was unmeasurable. Airplanes, birds and "
            "helicopters are hard negatives -- close enough to a drone "
            "that the model has to learn the difference."
        ),
        "why_separate_keys": (
            "train / validation / test are left untouched so every "
            "result produced before this addition stays comparable. An "
            "evaluation including negatives is train+train_neg and so "
            "on, which makes it a deliberate choice rather than a "
            "silent change of baseline."
        ),
        "caveat": (
            "1.5x more negative clips than positive. Feeding them all "
            "makes not detecting the cheapest way to cut the loss, so "
            "the share per batch has to be capped in train.py."
        ),
    }

    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote {splits_file}")
    print()
    print("Keys now:", ", ".join(sorted(splits)))
    print()
    print("train / validation / test are unchanged, so every existing")
    print("result still stands.")


if __name__ == "__main__":
    main()