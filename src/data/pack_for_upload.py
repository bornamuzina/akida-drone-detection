"""
Pack a tensor set into something small enough to upload.

Two savings, only one of which is lossy:

  compression  lossless. The tensors are ~85% zeros, which zip away to
               almost nothing. This is where most of the reduction comes
               from.

  float16      loses precision, but not usefully. Values run roughly
               -18 to +22 and float16 resolves to about 0.01 there. The
               training pipeline then clips to +/-5 and maps onto 256
               integer levels, so anything finer was going to be
               discarded regardless.

Measured on one clip: 60.4 MB -> 2.3 MB, about 26x.

Usage:
    python pack_for_upload.py
    python pack_for_upload.py --tensors ..\\..\\Data_new\\tensors_events\\tensors_messy_10ms
"""

from pathlib import Path
import argparse
import json
import shutil
import time

import numpy as np

# config lives in src/model, one directory across from this one. Adding
# it here means the script runs from anywhere without PYTHONPATH being
# set first.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", default=None,
                    help="tensor folder to pack (default: config.TENSOR_DIR)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: <tensors>_packed)")
    ap.add_argument("--dtype", default="uint8", choices=["uint8", "float16", "float32"])
    args = ap.parse_args()

    src = Path(args.tensors) if args.tensors else Path(config.TENSOR_DIR)
    dst = Path(args.out) if args.out else src.parent / f"{src.name}_packed"

    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    npys = sorted(src.glob("*_tensors.npy"))

    print(f"source : {src}")
    print(f"output : {dst}")
    print(f"clips  : {len(npys)}")
    print(f"dtype  : {args.dtype}")
    print()

    total_in = 0
    total_out = 0
    t0 = time.time()

    for i, npy in enumerate(npys, 1):
        clip = npy.stem[:-len("_tensors")]

        arr = np.load(npy, mmap_mode="r")
        arr = np.asarray(arr, dtype=np.dtype(args.dtype))

        out = dst / f"{clip}_tensors.npz"
        np.savez_compressed(out, a=arr)

        size_in = npy.stat().st_size
        size_out = out.stat().st_size

        total_in += size_in
        total_out += size_out

        # Metadata is small and holds the boxes, so it travels as-is.
        meta = src / f"{clip}_meta.json"
        if meta.exists():
            shutil.copy2(meta, dst / meta.name)
            total_out += meta.stat().st_size

        print(f"[{i:3d}/{len(npys)}] {clip}  "
              f"{size_in / 1e6:6.1f} -> {size_out / 1e6:5.2f} MB")

    # The split lists have to travel too, or the Colab side cannot
    # reproduce the same train/validation/test division.
    for extra in ("manifest.json",):
        f = src / extra
        if f.exists():
            shutil.copy2(f, dst / extra)

    splits = Path(config.SPLITS_FILE)
    if splits.exists():
        shutil.copy2(splits, dst / "splits.json")
        print()
        print("copied splits.json")
    else:
        print()
        print("WARNING: splits.json not found -- the Colab side will not")
        print("be able to reproduce the same split.")

    print()
    print(f"in     : {total_in / 1e9:.2f} GB")
    print(f"out    : {total_out / 1e9:.3f} GB")
    print(f"ratio  : {total_in / max(total_out, 1):.1f}x smaller")
    print(f"time   : {(time.time() - t0) / 60:.1f} min")
    print()
    print(f"Zip {dst.name} and upload it to Drive.")


if __name__ == "__main__":
    main()