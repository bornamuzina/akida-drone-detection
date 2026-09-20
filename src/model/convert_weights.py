"""
Rewrite a Keras 3 .weights.h5 into the Keras 2 (tf_keras) layout.

Why this exists
---------------
The 256 run was trained on Colab, where Keras 3 is the default. Keras 3
writes weights as

    /layers/<layer name>/vars/0, /vars/1, ...

Keras 2 -- which is what tf_keras uses, and tf_keras is what the Akida
toolchain pins -- expects

    /<layer name>   with a "weight_names" attribute

so tf_keras opens the Colab file, finds no variables where it looks, and
reports "expected 1 variables, but received 0". The weights are fine;
only the shelf they sit on is different.

The second problem
------------------
The Colab model's layers carry Keras' auto-generated names
("batch_normalization_7", "conv2d_3"), while the local model carries the
akida_models names ("conv_0/BN", "separable_12"). So the two files
cannot be matched up by name at all.

They can be matched by construction order. Keras' auto-naming is a
counter per layer class: the Nth Conv2D built in a session is called
"conv2d_<N-1>". So the Conv2D layers in the file, sorted by their
numeric suffix, line up one-for-one with the Conv2D layers of the local
model in the order it built them -- and likewise for every other class.

That is an inference, not a guarantee, so every single assignment is
shape-checked before it is made, and nothing is written unless all 49
agree. Two layers that should swap almost always have different channel
counts, so a wrong pairing shows up as a shape clash rather than
silently producing a model that scores badly.

Usage
-----
    python convert_weights.py --run yolov2_s16_rgb_256

Writes best.weights.k2.h5 next to the original. The original is not
touched.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import re

import h5py
import numpy as np

import config


def to_snake_case(name):
    """Keras' own class-name-to-layer-name rule, copied.

    Conv2D -> conv2d, BatchNormalization -> batch_normalization,
    SeparableConv2D -> separable_conv2d.
    """
    name = re.sub(r"\W+", "", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
    return name


def split_auto_name(key):
    """"batch_normalization_7" -> ("batch_normalization", 7)."""
    m = re.match(r"^(.*?)_(\d+)$", key)
    if m:
        return m.group(1), int(m.group(2))
    return key, 0


def keras3_layer_vars(f):
    """
    Map layer name -> list of arrays, read from a Keras 3 weights file.

    Returns None if the file is not in the Keras 3 layout, which is the
    signal that no conversion is needed.
    """
    if "layers" not in f:
        return None

    out = {}

    for name, grp in f["layers"].items():
        if "vars" not in grp:
            continue

        vars_grp = grp["vars"]

        # The keys are "0", "1", "2" ... and the order matters: it is the
        # order the layer declared its variables in. Sorting them as
        # strings would put "10" before "2", so sort numerically.
        keys = sorted(vars_grp.keys(), key=lambda k: int(k))
        arrays = [np.asarray(vars_grp[k]) for k in keys]

        if arrays:
            out[name] = arrays

    return out


def pair_up(layer_vars, model_layers):
    """
    Return [(model_layer, arrays, file_key)] or (None, reason).

    First tries an exact name match, which is what happens when both
    models were built by the same code. Falls back to matching within
    each layer class by construction order.
    """
    if all(l.name in layer_vars for l in model_layers):
        print("matching   : by layer name")
        return [(l, layer_vars[l.name], l.name) for l in model_layers], None

    print("matching   : by class and construction order")
    print("             (the file uses Keras' auto-generated names)")
    print()

    # ---- the file, grouped by class ---------------------------------
    by_family = defaultdict(list)
    for key in layer_vars:
        base, idx = split_auto_name(key)
        by_family[base].append((idx, key))

    for base in by_family:
        by_family[base].sort()

    # ---- the model, in the order it was built -----------------------
    model_families = defaultdict(list)
    for layer in model_layers:
        model_families[to_snake_case(type(layer).__name__)].append(layer)

    # ---- counts have to agree before anything is paired -------------
    all_families = set(by_family) | set(model_families)
    for base in sorted(all_families):
        n_file = len(by_family.get(base, []))
        n_model = len(model_families.get(base, []))
        flag = "" if n_file == n_model else "   <-- MISMATCH"
        print(f"  {base:<28} file {n_file:>3}   model {n_model:>3}{flag}")

    print()

    for base in sorted(all_families):
        if len(by_family.get(base, [])) != len(model_families.get(base, [])):
            return None, (
                f"class '{base}' appears a different number of times in "
                "the file than in the locally built model, so the two "
                "are not the same architecture"
            )

    pairs = []
    for base, layers in model_families.items():
        for layer, (_, key) in zip(layers, by_family[base]):
            pairs.append((layer, layer_vars[key], key))

    return pairs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="best.weights.h5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(config.RUNS_DIR) / args.run
    src = run_dir / args.weights

    if not src.exists():
        print(f"No weights at {src}")
        return

    dst = Path(args.out) if args.out else run_dir / "best.weights.k2.h5"

    print()
    print(f"source : {src}")
    print(f"target : {dst}")
    print()

    # ---- read the Keras 3 file -------------------------------------
    with h5py.File(src, "r") as f:
        layer_vars = keras3_layer_vars(f)

    if layer_vars is None:
        print("This file is already in the Keras 2 layout -- nothing to do.")
        print("If loading still fails, the problem is elsewhere.")
        return

    print(f"layers with weights in file : {len(layer_vars)}")

    # ---- build the model locally ------------------------------------
    from yolo_stride16 import yolo_base_stride16

    model = yolo_base_stride16(
        input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),
        classes=config.N_CLASSES,
        nb_box=config.N_ANCHORS,
        alpha=0.5,
    )

    model_layers = [l for l in model.layers if l.weights]
    print(f"layers with weights in model: {len(model_layers)}")
    print()

    pairs, reason = pair_up(layer_vars, model_layers)

    if pairs is None:
        print(f"Cannot pair the layers up: {reason}")
        print()
        print("Check INPUT_SIZE, N_ANCHORS and alpha in config.py against")
        print("what the Colab run used.")
        return

    # ---- shape check, before anything is assigned -------------------
    mismatched = []

    for layer, arrays, key in pairs:
        expected = [tuple(w.shape) for w in layer.weights]
        got = [tuple(a.shape) for a in arrays]
        if expected != got:
            mismatched.append((layer.name, key, expected, got))

    if mismatched:
        print(f"SHAPE CLASH: {len(mismatched)} of {len(pairs)} layers")
        for n, key, e, g in mismatched[:10]:
            print(f"  model '{n}'  <-  file '{key}'")
            print(f"    model expects {e}")
            print(f"    file has      {g}")
        if len(mismatched) > 10:
            print(f"  ... and {len(mismatched) - 10} more")
        print()
        print("Not writing the output. A partially loaded model would")
        print("score badly and look like a training problem rather than")
        print("a loading problem, which is the worst way to fail.")
        return

    print(f"shape check: all {len(pairs)} layers agree")

    # ---- assign ------------------------------------------------------
    for layer, arrays, _ in pairs:
        layer.set_weights(arrays)

    # ---- save the Keras 2 way ---------------------------------------
    model.save_weights(str(dst))

    print(f"wrote      : {dst}")

    # ---- prove it loads ---------------------------------------------
    fresh = yolo_base_stride16(
        input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),
        classes=config.N_CLASSES,
        nb_box=config.N_ANCHORS,
        alpha=0.5,
    )
    fresh.load_weights(str(dst))

    # Compare real arrays, not just "it did not throw". A file can load
    # cleanly and still be full of freshly initialised values.
    same = all(
        np.allclose(a, b)
        for layer in model_layers
        for a, b in zip(layer.get_weights(),
                        fresh.get_layer(layer.name).get_weights())
    )

    print(f"reload     : {'OK' if same else 'VALUES DIFFER -- do not use'}")

    if same:
        print()
        print("The pairing was inferred from construction order, so the")
        print("real proof is the score. Run the error analysis with")
        print("--weights best.weights.k2.h5 -- it prints AP as it goes.")
        print("Validation AP should land near 0.580. If it comes out")
        print("near zero, the pairing was wrong and nothing else is.")


if __name__ == "__main__":
    main()