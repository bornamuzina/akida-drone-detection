"""
Where in the frame do the training drones actually sit?

Why this exists
---------------
On test, the lower third of the frame gives 131 false positives against
416 outright misses -- against terrain the model does not find the
drone at all, rather than finding it and fumbling the box. The working
theory is that it learned a shortcut, "small dark thing on a smooth
background", which does not fire when the background is already full of
small dark things.

That theory is only worth acting on if the training data is actually
short of such frames. If a good share of training frames already put
the drone low and the model still fails there, the problem is not the
amount of data and a synthetic dataset will not fix it.

So this counts, per split, how many frames put the drone in the upper,
middle and lower band -- the same bands error_analysis.py reports
errors in, so the two tables can be read side by side.

The letterbox padding is skipped, since those rows are not image.

No model needed. Runs in seconds.

Usage
-----
    python band_counts.py
    python band_counts.py --split train
"""

from pathlib import Path
import argparse

import numpy as np

# This script lives one level down from the model code it imports, so
# put the parent directory on the path before importing any of it.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import dataset


def band_of(cy):
    """Which third of the real (unpadded) image a centre falls in."""
    pad = getattr(config, "PAD_Y", 0)
    top = pad
    bottom = config.INPUT_SIZE - pad
    height = max(bottom - top, 1)

    frac = (cy - top) / height

    if frac < 1 / 3:
        return "upper (sky)"
    if frac < 2 / 3:
        return "middle"
    return "lower (ground)"


def count_split(clips):
    """Count boxes per band, and collect their sizes."""
    totals = {"upper (sky)": 0, "middle": 0, "lower (ground)": 0}
    sizes = {"upper (sky)": [], "middle": [], "lower (ground)": []}
    per_clip = {}

    for clip in clips:
        _, records = dataset.load_clip(clip)

        if records is None:
            continue

        counts = {"upper (sky)": 0, "middle": 0, "lower (ground)": 0}

        for rec in records:
            for (x, y, w, h) in (rec.get("boxes") or []):
                b = band_of(y + h / 2)
                counts[b] += 1
                totals[b] += 1
                # sqrt(area) rather than width: it is the number the
                # recall-by-size table in evaluate.py uses, so the two
                # can be read together.
                sizes[b].append(float(np.sqrt(max(w * h, 0.0))))

        if sum(counts.values()):
            per_clip[clip] = counts

    return totals, sizes, per_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all",
                    help="train, validation, test, or all")
    args = ap.parse_args()

    splits = dataset.load_splits()

    names = (["train", "validation", "test"] if args.split == "all"
             else [args.split])

    print()
    print(f"tensors : {config.TENSOR_DIR}")
    print(f"padding : PAD_Y = {getattr(config, 'PAD_Y', 0)} px skipped "
          "at top and bottom")
    print()

    for name in names:
        totals, sizes, per_clip = count_split(splits[name])
        n = sum(totals.values())

        print("=" * 58)
        print(f"{name.upper()}   ({len(per_clip)} clips, {n} boxes)")
        print()

        if not n:
            print("  nothing loaded")
            print()
            continue

        print(f"  {'band':<16}{'boxes':>8}{'share':>8}"
              f"{'median':>9}{'p25':>7}{'p75':>7}{'under 8px':>11}")

        for b in ("upper (sky)", "middle", "lower (ground)"):
            s = np.array(sizes[b])

            if not len(s):
                print(f"  {b:<16}{0:>8}{0.0:>7.1f}%")
                continue

            print(f"  {b:<16}{totals[b]:>8}{totals[b] / n * 100:>7.1f}%"
                  f"{np.median(s):>8.1f}px{np.percentile(s, 25):>6.1f}"
                  f"{np.percentile(s, 75):>7.1f}"
                  f"{(s < 8).mean() * 100:>10.1f}%")

        # Clips are what you would go and look at, so name the ones
        # that actually carry the ground-band frames.
        ground = sorted(
            ((c, v["lower (ground)"], sum(v.values()))
             for c, v in per_clip.items() if v["lower (ground)"]),
            key=lambda r: -r[1],
        )

        print()
        if ground:
            print(f"  clips with drones low: {len(ground)} of {len(per_clip)}")
            for clip, low, tot in ground[:8]:
                print(f"    {clip:<20}{low:>6} of {tot:>5} "
                      f"({low / tot * 100:.0f}%)")
            if len(ground) > 8:
                print(f"    ... and {len(ground) - 8} more")
        else:
            print("  NO clip puts a drone in the lower band at all.")

        print()

    print("=" * 58)
    print("READING THIS")
    print()
    print("  Compare the train percentages with where the errors are.")
    print("  On test, the lower band holds 131 false positives and 416")
    print("  misses -- far more misses than anywhere else.")
    print()
    print("  If the training share for that band is small, the model")
    print("  has barely seen the case and more data is the right fix.")
    print()
    print("  If it is comparable to the other bands and the model still")
    print("  fails there, data quantity is not the problem, and a")
    print("  synthetic dataset would be solving the wrong thing.")
    print()
    print("  One caveat: the band is a proxy for background. A drone low")
    print("  in the frame over open water is not a cluttered background.")
    print("  If the counts look borderline, the clips named above are")
    print("  the ones to actually open.")


if __name__ == "__main__":
    main()