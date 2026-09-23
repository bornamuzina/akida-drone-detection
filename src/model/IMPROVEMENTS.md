# Improvements

The error analysis in `fail_reasons/` ended on one gap: not a single frame
in the dataset was free of drones, so nobody knew how often the model
cried wolf. These two cycles close that gap and build on it.

---

# Cycle 1 — negatives

Runs: `yolov2_s16_rgb_256` (the old model) and `yolov2_s16_rgb_256_neg`.

## What was done

**171 non-drone clips added as negatives.** The source dataset holds 59
airplane, 51 bird and 61 helicopter clips that had never been used. Their
own labels are ignored entirely -- the model stays single-class -- and
every frame carries an empty box list. They exist to teach it to stay
silent, which nothing had ever asked it to do.

They went in under their own keys in `splits.json`
(`train_neg` / `validation_neg` / `test_neg`), split 79/10.5/10.5 by clip
and stratified by class. The original `train` / `validation` / `test` keys
were left untouched, so every earlier number stays comparable.

One frame in three is now drone-free. `--negatives` on `train.py` and
`evaluate.py` turns them on; off is the old behaviour, exactly.

## What it showed

**The old model fired on 95.6% of drone-free frames.** It was not a good
detector. It was a loud one, measured on a set where being loud is never
punished.

Its headline AP of 0.602 was an artefact of that set. On the same test
split with drone-free frames included, it scores 0.428.

**Retraining with negatives changes the picture.** On test, comparing like
with like:

|                              | old   | with negatives |
| ---------------------------- | ----- | -------------- |
| AP@0.5                       | 0.428 | 0.563          |
| precision                    | 0.412 | 0.698          |
| false alarms per empty frame | 1.524 | 0.016          |

False alarms fell by roughly 95x.

The cost lands on small targets: recall drops ten points overall and
collapses on drones under 8 px. Two reasons, both plausible. The evidence
needed before firing has risen, and small targets produce the weakest
evidence. And the negatives include 51 bird clips, where a bird at 8 px
looks very much like a drone at 8 px.

Worth noting for anyone reading the old numbers: on the drones-only metric
the new model looks _worse_. Only the negatives-inclusive measurement shows
the improvement. Measured the old way, this change would have been
rejected.

## Loss is not the metric

The run also exposed a problem with how a checkpoint is chosen.

Validation loss is a per-slot regression figure summed over 16x16x5 slots
per image, of which at most one holds a drone. With a third of frames
empty, the no-object term dominates it. A model can lower that figure
simply by becoming quieter, while finding fewer drones -- and that is what
happened: loss kept improving to the last epoch while detection peaked two
epochs earlier, and those weights were gone.

AP is the honest criterion -- decode to boxes, apply NMS, match by IoU,
take the area under the precision-recall curve. It cannot be trained on:
sorting, thresholding and matching have no gradient, which is why loss
exists in the first place. But nothing stops it being measured.

`train.py` now computes validation AP every epoch and keeps two sets of
weights:

```
best.weights.h5      lowest validation loss
best_ap.weights.h5   highest validation AP
```

It costs no extra forward pass -- the validation pass already runs, and its
output is kept and decoded afterwards. `--no_ap` skips it.

---

# Cycle 2 — pretrained backbone, freezing, augmentation

Run: `yolov2_s16_rgb_256_neg_pre_frz_aug`. Three changes, applied together.

**ImageNet initialisation.** `pretrained.py` copies AkidaNet's ImageNet
weights into the backbone, leaving the YOLO head random. The backbone had
until now started from noise and had to invent edge detectors out of 90
clips.

**Freezing.** For the first two epochs the backbone is held fixed and only
the head trains. The head starts random, and its early gradients are large
enough to destroy the ImageNet filters before they are useful. Once the
head has settled, the backbone unfreezes at a tenth of the learning rate.

**Augmentation.** `augment.py` moves the boxes with the pixels: horizontal
flip, brightness, contrast, small shifts, and shrinking the frame to
manufacture smaller targets. No vertical flip and no rotation, because the
band analysis showed that sky above and terrain below is real structure.

## Result

|                              | negatives | + pretrain, freeze, augment |
| ---------------------------- | --------- | --------------------------- |
| AP@0.5                       | 0.563     | 0.607                       |
| recall                       | 0.610     | 0.724                       |
| false alarms per empty frame | 0.016     | 0.011                       |

Recall rose by eleven points while false alarms fell. Every earlier change
traded one against the other; this one moved both.

## What the run settled

**Loss was the better checkpoint criterion here.** The run kept both, and
the weights chosen by validation loss beat the ones chosen by validation
AP on test. Keeping two checkpoints costs nothing and removes the guess,
so the practice stays.

**Small targets did not improve.** Recall under 8 px is unchanged, despite
augmentation that shrinks frames specifically to produce them. Shrinking an
already-blurred 11 px drone does not make a realistic 6 px drone, it makes
mush. This gap will not close from the existing footage, and it is the
strongest argument for the synthetic dataset.

---

# Conclusions

The model was never short of drones. It was short of examples of what a
drone is not, and the metric it was judged by could not see the difference.
Fixing that was the single largest gain in the project, and it cost no new
data -- the clips were sitting in the source dataset, unused.

What followed -- a pretrained backbone, held still until the head caught
up, and augmentation -- added a second, smaller gain on top of it, without
giving back any of the first.

What remains is the drone that is only a few pixels across. Nothing done to
the training procedure has moved it, and nothing in the current footage can.
