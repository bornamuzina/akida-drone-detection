import json, glob
import numpy as np

import os
from pathlib import Path
# Resolve the project root from this file's location, so these scripts
# work from a clone rather than only from one machine. DRONE_ROOT
# overrides it when the data lives somewhere else.
ROOT = Path(os.environ.get("DRONE_ROOT", Path(__file__).resolve().parents[2]))

base = str(ROOT / "Data_new" / "tensors_events" / "tensors_messy_40ms_packed")

rows = []
for f in sorted(glob.glob(rf"{base}\*_meta.json")):
    m = json.load(open(f))
    W, H = m["input_resolution"]
    ev, inbox, area, n = [], 0, 0.0, 0
    for fr in m["frames"]:
        if fr["n_events_window"] == 0 or not fr["boxes"]:
            continue
        ev.append(fr["n_events_window"])
        inbox += fr["n_events_in_boxes"]
        area += sum(b[2] * b[3] for b in fr["boxes"]) / (W * H)
        n += 1
    if not n:
        continue
    ratio = (inbox / sum(ev)) / (area / n)
    rows.append((m["clip"], ratio, float(np.mean(ev)), float(np.median(ev))))

rows.sort(key=lambda r: r[1])

print(f"{'clip':<16}{'ratio':>8}{'mean_ev':>12}{'median_ev':>12}")
print("-" * 48)
print("WEAKEST 10")
for c, r, mn, md in rows[:10]:
    print(f"{c:<16}{r:>7.1f}x{mn:>12,.0f}{md:>12,.0f}")
print("\nSTRONGEST 10")
for c, r, mn, md in rows[-10:]:
    print(f"{c:<16}{r:>7.1f}x{mn:>12,.0f}{md:>12,.0f}")

weak = [r[2] for r in rows[:10]]
strong = [r[2] for r in rows[-10:]]
allm = [r[2] for r in rows]
print("\n" + "-" * 48)
print(f"weak   mean events/frame : {np.mean(weak):,.0f}")
print(f"strong mean events/frame : {np.mean(strong):,.0f}")
print(f"all    mean events/frame : {np.mean(allm):,.0f}")
print(f"\nweak/strong volume ratio : {np.mean(weak)/np.mean(strong):.2f}x")

r = np.corrcoef([x[1] for x in rows], [x[2] for x in rows])[0, 1]
print(f"corr(ratio, events/frame): {r:+.2f}")