# Where the model fails

Error analysis of `yolov2_s16_rgb_256` (stride-16 YOLOv2, 256x256 RGB),
run on validation and test, 2026-09-20.

Baseline: val 2674 TP / 1297 FP / 1054 FN, AP@0.5 0.580.
Test 3010 / 1251 / 1233, AP@0.5 0.602.

---

### Most false positives are the drone, found, with a loose box

77% of validation false positives (69% on test) sit on the real drone
but overlap it by less than half, so they do not count. 993 of them are
the same events as 993 of the 1054 false negatives.

**One mistake gets counted twice -- once as a miss, once as a false
alarm. The model does neither in these frames. It finds the drone and
draws the box slightly wrong.**

_Report the breakdown, not raw FP/FN, and give AP at 0.3 and 0.4 too._

### The boxes are off by about 2 pixels

Near misses overlap at 0.400 on validation and 0.410 on test. At a 0.3
threshold, 82% and 93% of them would count as hits.

**On a 10 pixel drone, being 2 pixels out is enough to fail. The number
is the same on both splits, so this is the model's real behaviour.**

_Track the median overlap as its own number, not only AP._

### The error is the model's, and it is scatter

Labels are steady between frames (0.79 px, 2.4%), and the
shakiest-labelled clip is one of the best performing -- so labels are
not the limit. The model's boxes are right on average but scatter from
20% too small to 15% too big, and correcting by the average gained 3
detections out of 2674.

**Not relabelling, not calibration. The model is imprecise per
detection.**

_Retrain with a DIoU loss. The current one compares widths as a squared
difference, which goes almost to zero on small objects, so precision on
10 px drones was never pushed for._

### The input is too small to do much better

At 256 a drone is about 11 px, roughly 120 pixels of evidence, and the
stride-16 backbone averages it down into about one cell. Asking for
2 px of precision from that is asking for an eighth of a cell.

**This is why 448 gained +0.218 AP while a finer grid at 224 gained
only +0.028. A bigger input and a finer grid are not the same thing.**

_AKD1500 caps input at 256x256x3, so this cannot be solved on this
chip. A better loss will help, but it cannot recover detail that the
downsampling already threw away._

### Low against terrain, the drone is missed completely

Test, lower third of the frame: 131 false positives against 416 misses
-- the reverse of the sky band. Validation does not show this (262 vs
130), so it depends on the clutter.

**Against sky the model finds the drone and fumbles the box. Against
trees and rooftops it does not find it at all.**

_This is the one failure more data clearly fixes. A synthetic dataset
should favour drones low against treelines, rooftops and hillsides
rather than more sky._

### There is no footage without a drone in it

Zero false positives on empty frames -- because every frame in both
splits contains a drone.

**We do not know how often the model cries wolf at an empty sky,
because it has never been shown one. In real use that is what it would
see almost all the time.**

_Convert the 171 non-drone clips already in the source dataset and
measure it. Include drone-free sequences in the synthetic dataset._

---

## Scripts

All three sit one level below the model code and add the parent folder
to the import path themselves.

| script              | what it answers                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `error_analysis.py` | what kind of error each FP and FN is; writes crops, contact sheets, a per-clip table and repeat-offender clusters |
| `label_jitter.py`   | whether the labels themselves are steady; no model needed, runs in seconds                                        |
| `box_bias.py`       | whether the box error is a fixed offset; measures it, applies it and re-scores                                    |

```
python fail_reasons/error_analysis.py --run yolov2_s16_rgb_256 \
    --split validation --weights best.weights.k2.h5
python fail_reasons/label_jitter.py --split validation
python fail_reasons/box_bias.py --run yolov2_s16_rgb_256 \
    --split validation --weights best.weights.k2.h5
```

Output goes to `runs/<run>/errors_<split>/`.

`best.weights.k2.h5` is the Keras 2 copy written by
`../convert_weights.py`. Colab saves Keras 3, which `tf_keras` cannot
read.

---

## Conclusion

The model finds drones reliably. Its boxes are about 2 px loose, which
the 0.5 IoU threshold punishes twice -- once as a miss, once as a false
alarm. It has never been tested on empty sky, and it misses drones
outright against ground clutter.

## Things to do

1. Measure the false-alarm rate on the 171 non-drone clips.
2. Retrain with a DIoU box loss, everything else identical.
3. Synthetic dataset: drones low against terrain, plus drone-free
   footage.
4. Report AP at 0.3 and 0.4 alongside 0.5.
