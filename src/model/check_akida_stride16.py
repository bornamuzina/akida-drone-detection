"""
Does the stride-16 YOLOv2 convert and map on Akida?

AKD1500 caps input at 256x256x3, so the 448 model that gave test AP
0.711 cannot be deployed. yolo_stride16.py gets the same 14x14 grid at
224 input instead, by removing the stride from separable_12.

That trade moves the cost from input size to activation size. The two
1024-filter tail layers now hold 14x14x1024 = 200,704 values each
instead of 7x7x1024 = 50,176. Weights are untouched -- both models have
3,568,590 parameters -- so if anything fails it will fail at convert or
map, not at build.

Both models are run so the comparison is like for like, on the same
machine with the same toolchain version.

Important limit: with no board attached everything maps to
BackendType.Software. That proves the graph is convertible, not that it
fits real AKD1500 memory. Necessary, not sufficient.

Run from the folder containing yolo_stride16.py.

Usage:
    python check_akida_stride16.py
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from akida_models import yolo_base
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert

from yolo_stride16 import yolo_base_stride16


INPUT_SIZE = 256     # the AKD1500 limit is 256; 224 keeps existing tensors
CHANNELS = 3         # RGB
CLASSES = 1          # drone
ANCHORS = 5
ALPHA = 0.5

VARIANTS = (
    ("stock stride 32", yolo_base),
    ("stride 16", yolo_base_stride16),
)


def try_variant(label, builder):
    result = {"label": label, "stage": None, "error": None}

    # ---- 1. build -------------------------------------------------
    try:
        model = builder(
            input_shape=(INPUT_SIZE, INPUT_SIZE, CHANNELS),
            classes=CLASSES,
            nb_box=ANCHORS,
            alpha=ALPHA,
        )
        result["keras_output"] = model.output_shape
        result["params"] = model.count_params()
        result["stage"] = "built"
    except Exception as e:
        result["stage"] = "FAILED at build"
        result["error"] = str(e)
        return result

    # ---- 2. quantize ----------------------------------------------
    try:
        qparams = QuantizationParams(
            input_weight_bits=8,
            weight_bits=4,
            activation_bits=4,
        )

        # Random calibration data. Meaningless numerically, but
        # quantization needs something to observe activation ranges.
        samples = np.random.rand(
            8, INPUT_SIZE, INPUT_SIZE, CHANNELS
        ).astype(np.float32)

        model_q = quantize(model, qparams=qparams, samples=samples,
                           num_samples=8, epochs=1)
        result["stage"] = "quantized"
    except Exception as e:
        result["stage"] = "FAILED at quantize"
        result["error"] = str(e)
        return result

    # ---- 3. convert to Akida ---------------------------------------
    try:
        model_akida = convert(model_q)
        result["stage"] = "converted"
        result["akida_layers"] = len(model_akida.layers)
    except Exception as e:
        result["stage"] = "FAILED at convert"
        result["error"] = str(e)
        return result

    # ---- 4. how does it map? ---------------------------------------
    # A model split across several sequences is being scheduled in
    # pieces, which usually means something did not fit. One sequence
    # is the healthy answer.
    try:
        seqs = model_akida.sequences
        result["sequences"] = len(seqs)
        result["backends"] = [str(s.backend) for s in seqs]
        result["stage"] = "mapped"
    except Exception as e:
        result["stage"] = "converted (mapping check failed)"
        result["error"] = str(e)

    # ---- 5. summary, best effort -----------------------------------
    try:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model_akida.summary()
        result["summary"] = buf.getvalue()
    except Exception:
        pass

    return result


def main():
    print()
    print("Stock vs stride-16 YOLOv2 against the Akida toolchain.")
    print(f"{INPUT_SIZE}x{INPUT_SIZE}x{CHANNELS}, {CLASSES} class, "
          f"{ANCHORS} anchors, alpha {ALPHA}.")
    print("=" * 68)

    results = []

    for label, builder in VARIANTS:
        print(f"\n--- {label} ---")
        r = try_variant(label, builder)
        results.append(r)

        print(f"  reached: {r['stage']}")

        if "keras_output" in r:
            print(f"  keras output shape : {r['keras_output']}")
            print(f"  parameters         : {r['params']:,}")

        if "akida_layers" in r:
            print(f"  akida layers       : {r['akida_layers']}")

        if "sequences" in r:
            print(f"  mapping sequences  : {r['sequences']}")
            for i, b in enumerate(r["backends"]):
                print(f"    sequence {i}: {b}")

        if r["error"]:
            lines = r["error"].strip().splitlines()
            print("  error:")
            for line in lines[:8]:
                print(f"    {line}")
            if len(lines) > 8:
                print(f"    ... ({len(lines) - 8} more lines)")

    # ---- verdict ------------------------------------------------------
    print()
    print("=" * 68)
    print("SUMMARY")
    print()

    for r in results:
        ok = r["stage"] in ("mapped", "converted")
        mark = "OK  " if ok else "FAIL"
        extra = ""
        if "sequences" in r and r["sequences"] > 1:
            extra = f"  (split into {r['sequences']} sequences)"
        grid = ""
        if "keras_output" in r:
            grid = f"  grid {r['keras_output'][1]}x{r['keras_output'][2]}"
        print(f"  {mark}  {r['label']:18s}{grid}  {r['stage']}{extra}")

    print()
    stock = results[0]
    s16 = results[1]

    if s16["stage"] != "mapped":
        print("Stride 16 does NOT map. The 14x14 grid is not reachable")
        print("this way. Remaining options: accept the 7x7 grid at 224,")
        print("or raise the input limit with the hardware team.")
    elif s16.get("sequences", 99) > stock.get("sequences", 1):
        print("Stride 16 maps but splits into more sequences than stock.")
        print("Something did not fit in one pass. Worth asking whether")
        print("that is acceptable before training on it.")
    else:
        print("Stride 16 maps as cleanly as stock, at the same parameter")
        print("count, within the 256 input limit. Structurally this is")
        print("the 14x14 grid without the 448 input.")
        print()
        print("Next: recompute anchors at DST=224, GRID=14 -- the cell is")
        print("now 16 px, not 32, so the existing values are wrong -- then")
        print("point train.py and evaluate.py at yolo_base_stride16.")

    print()
    print("IMPORTANT: with no board attached everything maps to")
    print("BackendType.Software. That proves the graph converts, not")
    print("that it fits AKD1500 on-chip memory. separable_12 and")
    print("separable_13 now hold 4x the activations they did at stride")
    print("32, and activation memory is the real constraint. Confirm on")
    print("hardware before treating this as settled.")

    if "summary" in s16:
        print()
        print("=" * 68)
        print("AKIDA MODEL SUMMARY (stride 16)")
        print(s16["summary"])


if __name__ == "__main__":
    main()