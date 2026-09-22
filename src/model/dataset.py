"""
Load event tensors and their boxes, split by clip.

The tensor stacks total several GB, so they are memory-mapped rather
than read into RAM. Only the frames actually being used get paged in.

Boxes are already in 224x224 space -- build_tensors.py applied the
letterbox transform when it wrote them -- so nothing is rescaled here.

Negatives
---------
Clips of airplanes, birds and helicopters carry frames with an empty box
list: their own class is ignored and they exist only to teach "nothing
here". They live in the same tensor folder as the drone clips and are
selected by name, through the *_neg keys in splits.json, so nothing
reaches training unless it was asked for.

Two things have to be switched on for them to be used at all:

    clips_for(splits, "train", negatives=True)   picks the clip names
    EventDataset(..., keep_empty=True)           indexes boxless frames

Both default to off, so every earlier result reproduces unchanged.
"""

from pathlib import Path
import json

import numpy as np

import config


# ============================================================
# LOADING
# ============================================================

def load_splits(path=None):
    """
    Read splits.json.

    The three original keys are always present. The _neg keys are
    returned as empty lists when absent, so this still works against a
    splits.json written before negatives existed.
    """
    path = Path(path or config.SPLITS_FILE)

    with open(path) as f:
        data = json.load(f)

    return {
        "train": data["train"],
        "validation": data["validation"],
        "test": data["test"],
        "train_neg": data.get("train_neg", []),
        "validation_neg": data.get("validation_neg", []),
        "test_neg": data.get("test_neg", []),
    }


def clips_for(splits, name, negatives=False):
    """
    The clip list for one split, optionally with its negatives.

    One place decides this, rather than every script writing
    splits["train"] + splits["train_neg"] and one of them forgetting.
    """
    clips = list(splits[name])

    if negatives:
        clips += splits.get(f"{name}_neg", [])

    return clips


def load_clip(clip, tensor_dir=None):
    """
    Load one clip's tensors and per-frame records.

    Returns (tensors, records) where tensors is a memory-mapped array of
    shape (N, 224, 224, 1) and records is the list of frame dicts from
    the metadata, in the same order.

    Returns (None, None) if the clip has not been generated. That is a
    normal state while conversion is still running, not an error.
    """
    tensor_dir = Path(tensor_dir or config.TENSOR_DIR)

    npy = tensor_dir / f"{clip}_tensors.npy"
    meta_path = tensor_dir / f"{clip}_meta.json"

    npz = tensor_dir / f"{clip}_tensors.npz"
    if not meta_path.exists() or (not npy.exists() and not npz.exists()):
        return None, None

    # mmap_mode='r' keeps the file on disk and pages in what is touched.
    # Loading 90 clips outright would be ~5.6 GB before TensorFlow has
    # allocated anything.
    tensors = np.load(npy, mmap_mode="r") if npy.exists() else np.load(npz)["a"]

    with open(meta_path) as f:
        meta = json.load(f)

    records = meta["frames"]

    if len(tensors) != len(records):
        raise ValueError(
            f"{clip}: {len(tensors)} tensors but {len(records)} records. "
            "The .npy and _meta.json are out of step -- regenerate the clip."
        )

    return tensors, records


# ============================================================
# INDEXING
# ============================================================

class EventDataset:
    """
    A flat, indexable view over many clips.

    Rather than concatenating every clip into one huge array, this keeps
    an index of (clip, position) pairs and reaches into the memory-mapped
    stacks on demand. Memory stays flat regardless of dataset size.
    """

    def __init__(self, clips, tensor_dir=None, quiet_policy=None,
                 verbose=True, keep_empty=False):
        self.tensor_dir = Path(tensor_dir or config.TENSOR_DIR)
        self.quiet_policy = quiet_policy or config.QUIET_POLICY

        # Frames with no box used to be dropped unconditionally, since
        # in the drone-only dataset a boxless frame meant a labelling
        # gap. Now it can also mean a deliberate negative, so it is a
        # choice -- and the default is the old behaviour.
        self.keep_empty = keep_empty

        self._tensors = {}      # clip -> memmapped array
        self._records = {}      # clip -> list of frame dicts

        self.index = []         # (clip, position within that clip)

        self.missing = []
        self.n_quiet = 0
        self.n_dropped = 0
        self.n_negative = 0

        for clip in clips:
            tensors, records = load_clip(clip, self.tensor_dir)

            if tensors is None:
                self.missing.append(clip)
                continue

            self._tensors[clip] = tensors
            self._records[clip] = records

            for i, rec in enumerate(records):
                if not rec["boxes"]:
                    if not self.keep_empty:
                        continue
                    self.n_negative += 1

                if rec["quiet"]:
                    self.n_quiet += 1
                    if self.quiet_policy == "drop":
                        self.n_dropped += 1
                        continue

                self.index.append((clip, i))

        if verbose:
            self.summary()

    # ---- basics -------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        """
        Returns (image, boxes, record).

        image  : float32 (224, 224, 1)
        boxes  : float32 (n, 4) as (x, y, w, h) in 224-space pixels,
                 and (0, 4) for a negative frame
        record : the frame's metadata dict
        """
        clip, pos = self.index[i]

        # np.asarray forces the memory-mapped slice into a real array.
        image = np.asarray(self._tensors[clip][pos], dtype=np.float32)

        rec = self._records[clip][pos]
        boxes = np.asarray(rec["boxes"], dtype=np.float32).reshape(-1, 4)

        return image, boxes, rec

    # ---- reporting ----------------------------------------------

    def summary(self):
        print(f"  clips loaded : {len(self._tensors)}")

        if self.missing:
            print(f"  MISSING      : {len(self.missing)} clips not generated yet")
            for c in self.missing[:5]:
                print(f"                 {c}")
            if len(self.missing) > 5:
                print(f"                 ... and {len(self.missing) - 5} more")

        print(f"  samples      : {len(self.index)}")

        if self.keep_empty:
            share = self.n_negative / max(len(self.index), 1) * 100
            print(f"  negatives    : {self.n_negative} ({share:.1f}% of samples)")
        elif self.n_negative:
            print(f"  negatives    : {self.n_negative} dropped "
                  "(keep_empty is off)")

        print(f"  quiet frames : {self.n_quiet} "
              f"(policy: {self.quiet_policy}"
              f"{f', {self.n_dropped} dropped' if self.n_dropped else ''})")

        if not self.index:
            return

        n_boxes = sum(
            len(self._records[c][i]["boxes"]) for c, i in self.index
        )
        print(f"  boxes        : {n_boxes}")

    def box_stats(self):
        """Size distribution of the boxes actually in this split."""
        sizes = []
        for clip, pos in self.index:
            for (x, y, w, h) in self._records[clip][pos]["boxes"]:
                sizes.append(np.sqrt(w * h))

        if not sizes:
            return None

        s = np.array(sizes)
        return {
            "n": len(s),
            "median_px": float(np.median(s)),
            "median_cells": float(np.median(s) / config.CELL),
            "p5": float(np.percentile(s, 5)),
            "p95": float(np.percentile(s, 95)),
            "under_8px": float((s < 8).mean()),
        }


# ============================================================
# BATCHING
# ============================================================

def batches(dataset, batch_size=None, shuffle=True, seed=None):
    """
    Yield (images, boxes_list) batches.

    Images stack cleanly into an array. Boxes cannot -- frames have
    different numbers of drones -- so they stay a list, and the target
    builder is what turns them into a fixed-shape tensor. A negative
    frame contributes an empty (0, 4) array, which make_target turns
    into an all-zero target.
    """
    batch_size = batch_size or config.BATCH_SIZE

    order = np.arange(len(dataset))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)

    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]

        images = []
        boxes_list = []

        for i in chunk:
            image, boxes, _ = dataset[i]
            images.append(image)
            boxes_list.append(boxes)

        yield np.stack(images), boxes_list


# ============================================================
# CHECK
# ============================================================

def main():
    """Load every split and report. Run this to verify the data is sane."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", action="store_true",
                    help="include the non-drone clips as negatives")
    args = ap.parse_args()

    splits = load_splits()

    print(f"tensors : {config.TENSOR_DIR}")
    print(f"policy  : {config.QUIET_POLICY}")
    print(f"negatives: {'on' if args.negatives else 'off'}")
    print()

    datasets = {}

    for name in ("train", "validation", "test"):
        clips = clips_for(splits, name, args.negatives)

        print(f"{name.upper()}  ({len(clips)} clips in split)")
        ds = EventDataset(clips, keep_empty=args.negatives)
        datasets[name] = ds

        stats = ds.box_stats()
        if stats:
            print(f"  box size     : median {stats['median_px']:.1f} px "
                  f"({stats['median_cells']:.2f} cells), "
                  f"p5 {stats['p5']:.1f}, p95 {stats['p95']:.1f}")
            print(f"  under 8 px   : {stats['under_8px'] * 100:.1f}%")
        print()

    # ---- inspect one batch -----------------------------------------
    train = datasets["train"]

    if len(train) == 0:
        print("No training samples yet -- conversion is probably still running.")
        return

    images, boxes_list = next(batches(train, batch_size=4, seed=0))

    print("One batch:")
    print(f"  images shape : {images.shape}  {images.dtype}")
    print(f"  value range  : {images.min():.2f} to {images.max():.2f}")
    print(f"  zero pixels  : {(images == 0).mean() * 100:.1f}%")
    print(f"  boxes/frame  : {[len(b) for b in boxes_list]}")

    # A box outside the image, or one landing in the letterbox padding,
    # means the geometry disagrees somewhere.
    bad = 0
    in_pad = 0
    for boxes in boxes_list:
        for (x, y, w, h) in boxes:
            if x < 0 or y < 0 or x + w > config.INPUT_SIZE or y + h > config.INPUT_SIZE:
                bad += 1
            if y + h <= config.PAD_Y or y >= config.INPUT_SIZE - config.PAD_Y:
                in_pad += 1

    print(f"  out of bounds: {bad}")
    print(f"  inside padding: {in_pad}")

    if bad or in_pad:
        print("  Geometry problem -- check the letterbox transform.")


if __name__ == "__main__":
    main()