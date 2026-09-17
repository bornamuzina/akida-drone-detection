# Drone detection for Akida

Detecting small drones in sky footage, targeting BrainChip's AKD1500
neuromorphic chip. Two strands: converting ordinary video into
event-camera data, and training detectors on both that and ordinary RGB
so the two can be compared.

---

## The headline result

Six YOLOv2 configurations and one modern baseline, all on the same 114
clips with the same clip-level splits.

| run | input | grid | drone at median | test AP@0.5 | deployable |
|---|---|---|---|---|---|
| `yolov2_rgb_224` | 224 | 7×7 | 10.4 px | 0.288 | yes |
| `yolov2_s16_rgb_224` | 224 | 14×14 | 10.4 px | 0.502 | yes |
| **`yolov2_s16_rgb_256`** | **256** | **16×16** | **11.9 px** | **0.602** | **yes** |
| `yolov2_rgb_448` | 448 | 14×14 | 20.7 px | 0.711 | no |
| `yolov2_rgb_448_aug` | 448 | 14×14 | 20.7 px | 0.634 | no |
| `yolo26n_rgb_224` | 224 | anchor-free | 10.4 px | 0.779 | not tested |

**Target size dominates everything else.** 10.4 → 11.9 → 20.7 px maps to
test AP 0.288 → 0.602 → 0.711. No other change came close.

**Grid resolution alone does not help.** `yolov2_s16_rgb_224` and
`yolov2_rgb_448` produce identical 14×14 grids with identical anchors in
cell units. The 448 version gained +0.218 validation AP over baseline;
the stride-16 version gained +0.028. Same grid, eight times the benefit —
because the pixels differ, not the grid. A finer grid cannot recover
detail that is not in the image.

**The best result is not deployable.** AKD1500 caps input at 256×256×3,
so 448 is out. `yolov2_s16_rgb_256` is the best configuration that fits:
it reaches 85% of the 448 result by removing one stride from the
backbone rather than enlarging the input.

**The architecture comparison is not like for like.** YOLO26n scores
higher with fewer parameters (2.38M vs 3.57M), but it had COCO
pretraining, three detection scales and real augmentation. The YOLOv2
side had none of those. Most of that gap is probably not architecture.

### Augmentation — inconclusive

Every other variable has a clear direction. This one does not. Flip and
brightness helped validation and hurt test; adding crop and cosine decay
was worse than plain 448 on both. The crop is the suspect — it only
zoomed in, so the model trained on a size distribution shifted toward
larger targets.

All three attempts were at 448, which is not deployable. Nothing has
been tried on the configuration that would actually ship.

---

## Hardware constraints

The chip is the reason several things look the way they do. Three facts
shaped the work:

**Input is capped at 256×256×3.** This is why the best-scoring model is
not the recommended one, and why the stride-16 backbone exists —
`src/model/yolo_stride16.py` removes the downsample from `separable_12`
to reach a 16×16 grid at 256, instead of enlarging the input to get
there.

**The input layer takes 1 or 3 channels, not 2.** A 2-channel polarity
histogram — the obvious event representation — fails at `cnn2snn`
convert with "Only signed inputs are supported". That is why the event
strand uses single-channel signed accumulation instead. RGB is
unaffected, being 3 channels already.

**Parameters do not change with input size** — 3,568,590 at every
resolution tested, because convolutional filters do not depend on
spatial extent. What grows is the activation maps, and activation memory
is the real constraint.

**Unverified:** every conversion check ran with no board attached, so
everything mapped to `BackendType.Software`. That proves the graph
converts, not that it fits on-chip memory.

---

## Layout

```
Data_raw/            source dataset, immutable, never written to
  Video_V/             285 visible-spectrum clips + .mat labels
  Video_IR/            365 infrared clips  (untouched — an open lead)
  Audio/               90 audio clips      (out of scope)

Data_new/            everything derived
  labels/              114 drone clips exported to frame,x,y,w,h CSV
  splits.json          the 90/12/12 clip-level division  ← load-bearing
  events/              simulated event streams, three presets
  tensors_events/      event tensors at 10/20/40 ms windows
  tensors_rgb/         RGB tensors at 224/256/448, plus the
                       jpg/txt tree Ultralytics reads
  runs/                one folder per trained model
  verification/        rendered frames behind the 20 ms decision
  annotated_rgb/       videos with boxes drawn on — see the warning in
                       its specs.md before using them for anything

src/
  data/                raw video → trainable tensors
  model/               train and evaluate
  checks/              verify, diagnose, answer questions

notebooks/             one per run — Colab or Kaggle, named in the file
requirements/          three environments, and why they cannot be merged
```

---

## The pipeline

Five stages. Each writes a folder the next one reads.

**1. Labels.** `Data_raw/Video_V/*_LABELS.mat` → `Data_new/labels/*.csv`

The `.mat` files hold MATLAB `groundTruth` MCOS objects, which scipy
cannot decode — the class definition that reconstructs them lives inside
the Computer Vision Toolbox. Exported once via MATLAB R2026a.
36,316 boxes across 114 drone clips. The 171 airplane / bird /
helicopter clips were skipped and remain available as background
negatives.

`Data_new/labels/label_extraction_notes.md` has the full story.
`mcos-decoder` on PyPI can now do this in pure Python, which would
remove the MATLAB dependency for anyone repeating it.

**2. Events** (event strand only). `src/data/convert_video_to_events.py`

Prophesee's own `viz_video_to_event_simulator.py` silently truncates at
50,000 events: its `-o` flag rides inside a display loop with a fixed
count buffer and there is no final drain. The replacement writes every
event and passes `--leak_rate_hz` through, which the original accepts
and then ignores. See `Data_new/tensors_events/event_truncation_bug.pdf`.

Three presets, parameters recorded in each folder's `specs.md`:

| preset | Cp/Cn | shot noise | used for |
|---|---|---|---|
| `clean` | 0.20 / 0.20 | 0 Hz | abandoned — see below |
| `default` | 0.15 / 0.10 | 10 Hz | one-clip diagnostic only |
| `messy` | 0.05 / 0.05 | 15 Hz | every event result |

`clean`'s thresholds proved too high for this footage — 25.7% of frames
came out with essentially no events, four clips with none at all.
Reconverting one clip at all three presets settled whether the fault was
the labels or the preset. `messy` dropped the empty-frame rate to 3.6%.

**3. Tensors.**

* RGB → `src/data/build_rgb_tensors.py`
* Events → `src/data/build_tensors.py` (one clip),
  `build_all_tensors.py` (all 114)

Letterboxed to the model input size — same scale on both axes, leftover
padded with zeros.

`build_rgb_tensors.py` writes two formats from the same pixels:

* `--format npy` → `.npy` stacks for `src/model/dataset.py`
* `--format yolo` → jpg + txt for Ultralytics

`src/data/pack_for_upload.py` compresses either to `.npz` for upload.
Event tensors shrink ~11× (85% zeros); RGB about 4×.

**4. Splits and anchors.** `src/data/make_splits.py`, `compute_anchors.py`

**Splits are by clip, never by frame.** 90 train / 12 validation / 12
test, stratified by median target size. At 30 fps consecutive frames are
near-duplicates, so a frame-level split would put near-identical images
on both sides and inflate every number in this repo.

Anchors come from IoU k-means over all 36,316 boxes, mean IoU 0.818.
They are expressed in **grid cells**, which is what `akida_models`
expects, so they must be recomputed whenever input size or stride
changes. `src/model/config.py` keeps all four sets, commented, with the
pixel sizes noted.

**5. Train and evaluate.** `src/model/train.py`, `evaluate.py`

Both read `config.py` for geometry and anchors, so target creation and
inference decoding cannot disagree. Training keeps the best epoch by
validation loss. Evaluation reports precision, recall, AP@0.5 and
recall broken down by target size.

---

## Things that would have failed silently

None of these raised an error.

**`yolo_base` normalises internally**, so inputs must be 0–255. Feed it
floats near ±1 and it learns nothing.

**Event tensors clip at ±5, not tighter.** Drone pixels saturate more
readily than background, so a tight clip destroys the signal.

**`to_uint8` tests values, not dtype.** `dataset.py` casts to float32
first, so a dtype check never fires. Cost one full run at recall 0.000.

**Judge augmentation on AP, not validation loss.** Loss only compares
epochs within a single run.

**Keras 2 and Keras 3 write H5 weights differently.** Colab saves in the
Keras 3 layout; loading locally through `tf_keras` reports "expected 1
variables, but received 0" — same weights, different filing system. Load
weights in the environment that wrote them, or use the `.keras` file,
which both versions read.

---

## Known limitations

* **One seed per run.** Differences under roughly 0.05 AP are not
  clearly outside run-to-run variance — which covers the augmentation
  results, but not the resolution ones.
* **Validation and test are not equally hard.** Medians match but tails
  do not — sub-8px boxes are 25.0% of validation and 4.6% of test. The
  two splits can rank runs differently, and did.
* **The largest anchor covers 1.5% of boxes** and is barely learned. In
  several runs recall on 16+ px targets is worse than on 12–16 px, which
  is backwards.
* **Frames were pre-letterboxed before Ultralytics resized them again**,
  so padding becomes trained content for YOLO26n. Deliberate — it keeps
  the pixels identical between the two models — but a run on raw
  640×512 frames would likely score higher.

---

## To-do ideas

* Augmentation at 256, since every attempt so far was at 448.
* Pretrained ImageNet weights — untried, and cheap: `yolo_stride16.py`
  keeps every stock AkidaNet layer name, so a checkpoint should load.
* Fix the crop to zoom out as well as in.
* The 365 infrared sequences in `Data_raw/Video_IR`, at 640×512 — the
  same resolution as the event sensor being simulated.
* The 171 non-drone clips as background negatives.
* More data. Drone-vs-Bird (WOSDETC) is the closest public match —
  static cameras, birds as distractors, near-identical annotation
  format; request access at wosdetc@googlegroups.com. Anti-UAV (MIT)
  has 318 RGB+IR video pairs from fixed ground cameras.

---

## Getting started

```powershell
# 1. Source data -- not in this repo. CC0, 2 GB.
#    https://doi.org/10.5281/zenodo.5500576
#    Extract to Data_raw/

# 2. Pick an environment. See requirements/README.md for why there are
#    three and why they cannot be merged.
python -m venv C:\tmp\akida\venv
C:\tmp\akida\venv\Scripts\Activate.ps1
pip install -r requirements\requirements-akida.txt

# 3. Build RGB tensors. Set DST in the script first -- 224, 256 or 448.
python src\data\build_rgb_tensors.py --format npy

# 4. Recompute anchors for that geometry, then paste them into
#    src\model\config.py along with INPUT_SIZE and GRID.
python src\data\compute_anchors.py --labels Data_new\labels

# 5. Check the data loads and the box statistics look right.
python src\model\dataset.py

# 6. Train. Locally this is impractical -- use a notebook from
#    notebooks/ on Colab or Kaggle with a T4.
python src\model\train.py --epochs 15 --lr 1e-3 --batch_size 64 --name my_run
python src\model\evaluate.py --run my_run --split validation
```

`--lr 1e-3` is not a default worth changing lightly: 1e-4 collapses the
model to predicting nothing, because with 0.4% of output slots positive
that is a strong local minimum.

---

## Source data

Halmstad **Drone-detection-dataset**, Fredrik Svanström et al.,
released **CC0 1.0** (public domain). 650 videos, 203,328 annotated
frames. DOI [10.5281/zenodo.5500576](https://doi.org/10.5281/zenodo.5500576).

Citation offered voluntarily:
Svanström F, Alonso-Fernandez F, Englund C. *A Dataset for Multi-Sensor
Drone Detection.* Data in Brief, 2021.

`Data_raw/README_source_dataset.md` is the dataset's own README, kept
as received.
