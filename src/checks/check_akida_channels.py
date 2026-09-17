"""
Does a YOLOv2 at N x N input actually convert to an Akida model?

Same shape as check_akida_channels.py, but the variable is input
resolution rather than channel count. 448 was adopted because it lifts
the median target from 0.33 to 0.65 grid cells, which nearly doubled
AP. That gain is worthless if the model cannot be deployed.

The weight count is identical at every resolution -- convolutions do
not care about input size. What changes is the size of the activation
maps flowing between layers, and that is what Akida's on-chip memory
constrains. So the interesting failure would appear at map, not at
build.

Important limit: with no board attached, everything maps to
BackendType.Software. That proves the graph is convertible, not that it
fits real AKD1500 memory. A Software backend here is a necessary
condition, not a sufficient one.

Usage:
    python check_akida_resolution.py
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from akida_models import yolo_base
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert


CHANNELS = 3         # RGB
CLASSES = 1          # drone
ANCHORS = 5

# 224 is the known-good reference. 448 is what we actually want.
# 320 sits between them, so a 448 failure can be bracketed rather than
# just reported.
SIZES = (224, 320, 448)


def try_size(size):
    result = {"size": size, "stage": None, "error": None}

    # ---- 1. build -------------------------------------------------
    try:
        model = yolo_base(
            input_shape=(size, size, CHANNELS),
            classes=CLASSES,
            nb_box=ANCHORS,
            alpha=0.5,
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
        samples = np.random.rand(8, size, size, CHANNELS).astype(np.float32)

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

    # ---- 5. memory summary, if the toolchain offers one -------------
    # This is the number that would actually decide AKD1500 fit. Not
    # every version exposes it, so it is best-effort.
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
    print("Testing YOLOv2 input resolution against the Akida toolchain.")
    print(f"{CHANNELS} channels, {CLASSES} class, {ANCHORS} anchors, alpha 0.5.")
    print("=" * 68)

    results = []

    for size in SIZES:
        print(f"\n--- {size} x {size} ---")
        r = try_size(size)
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
            for line in lines[:6]:
                print(f"    {line}")
            if len(lines) > 6:
                print(f"    ... ({len(lines) - 6} more lines)")

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
        print(f"  {mark}  {r['size']}x{r['size']}: {r['stage']}{extra}")

    print()
    big = next(r for r in results if r["size"] == 448)

    if big["stage"] == "mapped" and big.get("sequences", 99) == 1:
        print("448 converts and maps as a single sequence. The resolution")
        print("that gave AP 0.711 is structurally deployable.")
    elif big["stage"] in ("mapped", "converted"):
        print("448 converts but does not map cleanly. Check the sequence")
        print("count above -- more than one means it was split.")
    else:
        print("448 does NOT convert. The accuracy gain from higher")
        print("resolution is not deployable on this toolchain. Either")
        print("stay at 224, or raise it with the hardware team.")

    print()
    print("IMPORTANT: with no board attached everything maps to")
    print("BackendType.Software. That proves the graph converts, not")
    print("that it fits AKD1500 on-chip memory. Activation maps are 4x")
    print("larger at 448 than at 224 even though the weights are")
    print("identical, and activation memory is the real constraint.")
    print("Confirm on hardware before treating this as settled.")

    # Print the summary for the largest size that worked -- it is the
    # closest thing to a memory report the toolchain gives us.
    ok_results = [r for r in results if "summary" in r]
    if ok_results:
        last = ok_results[-1]
        print()
        print("=" * 68)
        print(f"AKIDA MODEL SUMMARY ({last['size']}x{last['size']})")
        print(last["summary"])


if __name__ == "__main__":
    main()