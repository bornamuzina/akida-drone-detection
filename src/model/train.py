"""
Train the YOLOv2 detector on event tensors.

Two things here are specific to this data and worth knowing:

1. Inputs are mapped to uint8 before the model sees them. yolo_base has
   a Rescaling layer expecting 0-255, and Akida's input layer is 8-bit,
   so this is required both for training to work at all and for the
   model to deploy. Float tensors would come out of Rescaling at
   approximately -1 everywhere and the model would learn nothing --
   silently, with no error.

2. The clip point is +/-5, not the tighter value that image intuition
   suggests. Drone pixels saturate roughly 150x more readily than
   background ones, because the drone is where events stack up. Clipping
   tight preferentially destroys the signal.

Negatives
---------
--negatives adds the airplane, bird and helicopter clips as frames with
no box. Watch max objectness when using it: loss_noobj grows with every
negative frame while the divisor n_obj does not, so "predict nothing"
becomes cheaper than it was when W_NOOBJ was chosen. If maxobj collapses
toward zero, lower W_NOOBJ before blaming anything else.

Usage:
    python train.py --clips 20 --epochs 10
    python train.py --epochs 50
    python train.py --epochs 50 --negatives --name run_neg
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import tensorflow as tf

import config
import dataset
import targets as T


CLIP = 5.0


# ============================================================
# INPUT SCALING
# ============================================================

def to_uint8(images, clip=CLIP):
    """
    Signed accumulation -> 0-255.

    Zero maps to 128, so 'nothing happened' sits at the midpoint and the
    two polarities occupy the halves either side of it.
    """
    if images.min() >= 0 and images.max() > 5:
        return images.astype(np.float32)
    x = np.clip(images, -clip, clip)
    return (x / clip * 127.0 + 128.0).astype(np.float32)


# ============================================================
# LOSS
# ============================================================
#
# Only 0.414% of the 245 output slots contain a drone. Without weighting,
# predicting "nothing" everywhere scores 99.6% and is a strong local
# minimum that detects nothing at all. The weights below are what stop
# the model settling there.

W_COORD = 5.0        # box geometry, only on slots that contain an object
W_OBJ = 5.0          # confidence where an object is
W_NOOBJ = 0.5        # confidence where there is not
W_CLASS = 1.0


def yolo_loss(y_true, y_pred):
    """
    y_true: (batch, 7, 7, 5, 6)  from targets.make_target
    y_pred: (batch, 7, 7, 30)    raw model output

    Predictions are raw. Sigmoids and exponentials are applied here so
    the network can output unbounded numbers and still produce valid
    boxes.
    """
    grid = config.GRID
    n_anchors = config.N_ANCHORS
    n_classes = config.N_CLASSES

    y_pred = tf.reshape(
        y_pred, (-1, grid, grid, n_anchors, 5 + n_classes)
    )

    anchors = tf.constant(config.ANCHORS, dtype=tf.float32)  # (5, 2)

    # ---- decode the prediction into cell units --------------------
    # Cell origin for each grid position, so offsets become absolute.
    cols = tf.range(grid, dtype=tf.float32)
    rows = tf.range(grid, dtype=tf.float32)
    gx, gy = tf.meshgrid(cols, rows)

    gx = tf.reshape(gx, (1, grid, grid, 1))
    gy = tf.reshape(gy, (1, grid, grid, 1))

    # Centre: sigmoid keeps it inside its own cell, then add the cell.
    pred_cx = tf.sigmoid(y_pred[..., 0]) + gx
    pred_cy = tf.sigmoid(y_pred[..., 1]) + gy

    # Size: exponential scales the anchor. Always positive, and a
    # prediction of 0 leaves the anchor unchanged.
    pred_w = tf.exp(tf.clip_by_value(y_pred[..., 2], -8, 8)) * anchors[:, 0]
    pred_h = tf.exp(tf.clip_by_value(y_pred[..., 3], -8, 8)) * anchors[:, 1]

    pred_obj = tf.sigmoid(y_pred[..., 4])
    pred_cls = tf.sigmoid(y_pred[..., 5:])

    # ---- ground truth ---------------------------------------------
    true_cx = y_true[..., 0]
    true_cy = y_true[..., 1]
    true_w = y_true[..., 2]
    true_h = y_true[..., 3]
    true_obj = y_true[..., 4]
    true_cls = y_true[..., 5:]

    mask = true_obj                          # 1 where a drone is

    n_obj = tf.maximum(tf.reduce_sum(mask), 1.0)

    # ---- coordinates ----------------------------------------------
    loss_xy = tf.reduce_sum(
        mask * (tf.square(pred_cx - true_cx) + tf.square(pred_cy - true_cy))
    )

    # Square roots on width and height so that a 2 px error on a small
    # box costs more than a 2 px error on a large one. With targets a
    # third of a cell wide this matters more than usual.
    loss_wh = tf.reduce_sum(
        mask * (
            tf.square(tf.sqrt(tf.maximum(pred_w, 1e-6))
                      - tf.sqrt(tf.maximum(true_w, 1e-6)))
            + tf.square(tf.sqrt(tf.maximum(pred_h, 1e-6))
                        - tf.sqrt(tf.maximum(true_h, 1e-6)))
        )
    )

    # ---- objectness ------------------------------------------------
    loss_obj = tf.reduce_sum(mask * tf.square(pred_obj - 1.0))
    loss_noobj = tf.reduce_sum((1.0 - mask) * tf.square(pred_obj - 0.0))

    # ---- class ------------------------------------------------------
    loss_cls = tf.reduce_sum(
        tf.expand_dims(mask, -1) * tf.square(pred_cls - true_cls)
    )

    total = (
        W_COORD * (loss_xy + loss_wh)
        + W_OBJ * loss_obj
        + W_NOOBJ * loss_noobj
        + W_CLASS * loss_cls
    )

    return total / n_obj


# ============================================================
# METRICS
# ============================================================

def batch_recall(y_true, y_pred, threshold=0.5):
    """
    Fraction of true objects whose own slot fired above threshold.

    Not mAP -- that needs decoding and NMS. This is a cheap signal during
    training for whether the model is predicting anything at all, which
    is the first thing that goes wrong.
    """
    y_pred = tf.reshape(
        y_pred, (-1, config.GRID, config.GRID, config.N_ANCHORS,
                 5 + config.N_CLASSES)
    )

    mask = y_true[..., 4]
    obj = tf.sigmoid(y_pred[..., 4])

    hit = tf.reduce_sum(mask * tf.cast(obj > threshold, tf.float32))

    return hit / tf.maximum(tf.reduce_sum(mask), 1.0)


def max_objectness(y_pred):
    """Highest confidence anywhere. If this stays near 0 the model has
    collapsed to predicting nothing."""
    y_pred = tf.reshape(
        y_pred, (-1, config.GRID, config.GRID, config.N_ANCHORS,
                 5 + config.N_CLASSES)
    )
    return tf.reduce_max(tf.sigmoid(y_pred[..., 4]))


# ============================================================
# TRAINING
# ============================================================

def run_epoch(model, ds, optimizer, batch_size, training=True, seed=None):
    losses = []
    recalls = []
    max_obj = 0.0

    for images, boxes_list in dataset.batches(
        ds, batch_size=batch_size, shuffle=training, seed=seed
    ):
        x = to_uint8(images)
        y = T.make_targets(boxes_list)

        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        if training:
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss = yolo_loss(y, pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            pred = model(x, training=False)
            loss = yolo_loss(y, pred)

        losses.append(float(loss))
        recalls.append(float(batch_recall(y, pred)))
        max_obj = max(max_obj, float(max_objectness(pred)))

    return float(np.mean(losses)), float(np.mean(recalls)), max_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=0,
                    help="use only the first N training clips (0 = all)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tensors", default=None,
                    help="override the tensor directory")
    ap.add_argument("--name", default="run")
    ap.add_argument("--negatives", action="store_true",
                    help="add the non-drone clips as boxless frames")
    args = ap.parse_args()

    # from akida_models import yolo_base
    from yolo_stride16 import yolo_base_stride16

    tensor_dir = Path(args.tensors) if args.tensors else config.TENSOR_DIR

    # ---- data --------------------------------------------------------
    splits = dataset.load_splits()

    train_clips = dataset.clips_for(splits, "train", args.negatives)
    if args.clips:
        train_clips = train_clips[:args.clips]

    print("TRAIN")
    train_ds = dataset.EventDataset(train_clips, tensor_dir=tensor_dir,
                                    keep_empty=args.negatives)
    print("VALIDATION")
    val_ds = dataset.EventDataset(
        dataset.clips_for(splits, "validation", args.negatives),
        tensor_dir=tensor_dir, keep_empty=args.negatives)

    if len(train_ds) == 0:
        raise SystemExit("No training samples.")

    # ---- model -------------------------------------------------------
    #model = yolo_base(input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),classes=config.N_CLASSES,nb_box=config.N_ANCHORS,alpha=args.alpha,)
    model = yolo_base_stride16(
        input_shape=(config.INPUT_SIZE, config.INPUT_SIZE, config.CHANNELS),
        classes=config.N_CLASSES,
        nb_box=config.N_ANCHORS,
        alpha=args.alpha,
    )
    print()
    print(f"model    : alpha {args.alpha}, {model.count_params():,} params")
    print(f"output   : {model.output_shape}")
    print(f"clip     : +/-{CLIP} -> uint8")
    print(f"lr       : {args.lr}, batch {args.batch_size}")
    print(f"negatives: {'on' if args.negatives else 'off'}")
    print()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    run_dir = Path(config.RUNS_DIR) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best = float("inf")

    print(f"{'epoch':>5}  {'train':>9}  {'val':>9}  "
          f"{'t.rec':>6}  {'v.rec':>6}  {'maxobj':>6}  {'min':>5}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_rec, _ = run_epoch(
            model, train_ds, optimizer, args.batch_size,
            training=True, seed=epoch
        )
        va_loss, va_rec, va_max = run_epoch(
            model, val_ds, optimizer, args.batch_size, training=False
        )

        dt = (time.time() - t0) / 60

        print(f"{epoch:5d}  {tr_loss:9.3f}  {va_loss:9.3f}  "
              f"{tr_rec:6.3f}  {va_rec:6.3f}  {va_max:6.3f}  {dt:5.1f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_recall": tr_rec,
            "val_recall": va_rec,
            "val_max_objectness": va_max,
            "minutes": dt,
        })

        # Keep the best by validation loss, not the last epoch.
        if va_loss < best:
            best = va_loss
            model.save_weights(str(run_dir / "best.weights.h5"))
            model.save(str(run_dir / "best.keras"))

        with open(run_dir / "history.json", "w") as f:
            json.dump({
                "args": vars(args),
                "clip": CLIP,
                "anchors": config.ANCHORS,
                "loss_weights": {
                    "coord": W_COORD, "obj": W_OBJ,
                    "noobj": W_NOOBJ, "cls": W_CLASS,
                },
                "negatives": args.negatives,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_train_negative": train_ds.n_negative,
                "n_val_negative": val_ds.n_negative,
                "history": history,
            }, f, indent=2)

        # The failure mode to watch for. If confidence never rises the
        # model has settled into predicting nothing everywhere, and the
        # no-object weight needs lowering. Adding negatives makes this
        # more likely, not less -- every boxless frame is pure noobj
        # loss and does not increase the n_obj it is divided by.
        if epoch >= 3 and va_max < 0.1:
            print()
            print("  Max objectness still near zero -- the model is")
            print("  predicting 'nothing' everywhere. Lower W_NOOBJ.")
            if args.negatives:
                print("  The negatives make this easier to fall into.")

    print()
    print(f"best val loss: {best:.3f}")
    print(f"weights      : {run_dir / 'best.weights.h5'}")


if __name__ == "__main__":
    main()