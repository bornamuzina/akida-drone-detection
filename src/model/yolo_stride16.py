"""
YOLOv2 with a 14x14 output grid at 224x224 input.

Why this exists
---------------
Raising input resolution to 448 gave the single largest accuracy gain
measured so far -- test AP 0.288 -> 0.711 -- by turning a 7x7 grid into
a 14x14 one and doubling the drone from 10 to 20 px. But AKD1500 caps
input at 256x256x3, so 448 is not deployable.

The grid is what actually mattered, and the grid comes from the number
of stride-2 layers, not from the input size. AkidaNet has five:

    conv_0        224 -> 112
    conv_2        112 ->  56
    separable_4    56 ->  28
    separable_6    28 ->  14
    separable_12   14 ->   7

Remove one and a 224 input produces 14x14 directly, within the Akida
input limit.

Which stride to remove
----------------------
separable_12, the last one. Only separable_12 and separable_13 then run
at 14x14 instead of 7x7; every layer before is bit-for-bit what the
current 224 model already does. Removing an earlier stride would double
the resolution of every layer after it, which is how the 448 attempt ran
into activation memory in the first place.

Cost: two layers of 1024 filters go from 7x7 to 14x14, so their
activations are 4x larger. 14x14x1024 is 200,704 values against 50,176.
Whether AKD1500 absorbs that is exactly what needs testing -- see
check_akida_resolution.py, adapted to this builder.

What does NOT change
--------------------
Parameter count. Convolutional filters do not depend on spatial size,
so this model has the same 3,568,590 weights at alpha 0.5 as the
stride-32 version. Only activations grow.

Anchors DO change. They are expressed in grid cells, and the cell is
now 16 px rather than 32, so recompute with GRID = 14 at DST = 224
before training.

Usage:
    from yolo_stride16 import yolo_base_stride16
    model = yolo_base_stride16(input_shape=(224, 224, 3),
                               classes=1, nb_box=5, alpha=0.5)
    print(model.output_shape)      # expect (None, 14, 14, 30)
"""

import numpy as np

import tensorflow as tf
from tf_keras import Model, regularizers
from tf_keras.layers import Input, Rescaling

from akida_models.layer_blocks import (conv_block, separable_conv_block,
                                       yolo_head_block)
from akida_models.utils import get_params_by_version


# The only line that differs from stock AkidaNet. Kept as a constant so
# the diff against the library is one value, not a rewritten file.
SEPARABLE_12_STRIDES = 1          # stock is 2


def akidanet_backbone_stride16(input_shape=(224, 224, 3),
                               alpha=1.0,
                               input_scaling=(127.5, -1)):
    """
    AkidaNet without the classifier head, and without the final stride.

    This is a transcription of akidanet_imagenet(include_top=False,
    pooling=None) from akida_models, with separable_12 changed from
    strides=2 to strides=1. Everything else -- filter counts, kernel
    sizes, padding, batchnorm, regularizer, activation -- is identical,
    so the layers keep their stock names and shapes and a stock
    checkpoint still matches them.
    """
    weight_regularizer = regularizers.l2(4e-5)

    fused, post_relu_gap, relu_activation = get_params_by_version(
        relu_v2='ReLU7.5')

    img_input = Input(shape=input_shape, name="input", dtype=tf.uint8)

    # The model expects 0-255 and rescales internally. This is why
    # to_uint8 has to hand it 0-255 rather than normalised floats.
    scale, offset = (1, 0) if input_scaling is None else input_scaling
    x = Rescaling(1. / scale, offset, name="rescaling")(img_input)

    # ---- 224 -> 112 --------------------------------------------------
    x = conv_block(x, filters=int(32 * alpha), name='conv_0',
                   kernel_size=(3, 3), padding='same', use_bias=False,
                   strides=2, add_batchnorm=True,
                   relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    x = conv_block(x, filters=int(64 * alpha), name='conv_1',
                   kernel_size=(3, 3), padding='same', use_bias=False,
                   add_batchnorm=True, relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    # ---- 112 -> 56 ---------------------------------------------------
    x = conv_block(x, filters=int(128 * alpha), name='conv_2',
                   kernel_size=(3, 3), padding='same', strides=2,
                   use_bias=False, add_batchnorm=True,
                   relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    x = conv_block(x, filters=int(128 * alpha), name='conv_3',
                   kernel_size=(3, 3), padding='same', use_bias=False,
                   add_batchnorm=True, relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    # ---- 56 -> 28 ----------------------------------------------------
    x = separable_conv_block(x, filters=int(256 * alpha),
                             name='separable_4', kernel_size=(3, 3),
                             padding='same', strides=2, use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation, fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x, filters=int(256 * alpha),
                             name='separable_5', kernel_size=(3, 3),
                             padding='same', use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation, fused=fused,
                             pointwise_regularizer=weight_regularizer)

    # ---- 28 -> 14 ----------------------------------------------------
    x = separable_conv_block(x, filters=int(512 * alpha),
                             name='separable_6', kernel_size=(3, 3),
                             padding='same', strides=2, use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation, fused=fused,
                             pointwise_regularizer=weight_regularizer)

    for i in (7, 8, 9, 10, 11):
        x = separable_conv_block(x, filters=int(512 * alpha),
                                 name=f'separable_{i}', kernel_size=(3, 3),
                                 padding='same', use_bias=False,
                                 add_batchnorm=True,
                                 relu_activation=relu_activation,
                                 fused=fused,
                                 pointwise_regularizer=weight_regularizer)

    # ---- THE CHANGE: 14 -> 14 instead of 14 -> 7 ---------------------
    x = separable_conv_block(x, filters=int(1024 * alpha),
                             name='separable_12', kernel_size=(3, 3),
                             padding='same',
                             strides=SEPARABLE_12_STRIDES,
                             use_bias=False, add_batchnorm=True,
                             relu_activation=relu_activation, fused=fused,
                             pointwise_regularizer=weight_regularizer)

    # No pooling: include_top is False and pooling is None, matching how
    # yolo_base calls the stock backbone.
    x = separable_conv_block(x, filters=int(1024 * alpha),
                             name='separable_13', kernel_size=(3, 3),
                             padding='same', pooling=None, use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation, fused=fused,
                             post_relu_gap=post_relu_gap,
                             pointwise_regularizer=weight_regularizer)

    return Model(img_input, x,
                 name='akidanet_s16_%0.2f_%s' % (alpha, input_shape[0]))


def yolo_base_stride16(input_shape=(224, 224, 3), classes=1, nb_box=5,
                       alpha=1.0, input_scaling=(127.5, -1)):
    """
    Same contract as akida_models.yolo_base, with a 14x14 output at 224.

    The detection-layer weight initialisation is copied from yolo_base:
    scaling the initial weights by 1/grid_area keeps early predictions
    small, which matters when only 0.4% of output slots contain an
    object and the model can otherwise settle into predicting nothing.
    """
    base_model = akidanet_backbone_stride16(input_shape=input_shape,
                                            alpha=alpha,
                                            input_scaling=input_scaling)

    x = yolo_head_block(base_model.layers[-1].output,
                        num_boxes=nb_box, classes=classes)
    model = Model(inputs=base_model.input, outputs=x,
                  name='yolo_base_stride16')

    # ---- detection layer init (lifted from yolo_base) ----------------
    layers = [l for l in model.layers if "detection_layer" in l.name]
    assert len(layers) in (1, 2), "No detection layer found."

    if len(layers) == 1:
        # sepconv is fused on Akida v1
        layer = layers[0]
        w = layer.get_weights()
        dw_shape, pw_shape, bias_shape = w[0].shape, w[1].shape, w[2].shape
    else:
        # sepconv is unfused on Akida v2
        dw_layer, pw_layer = layers[0], layers[1]
        dw_w = dw_layer.get_weights()
        assert len(dw_w) == 1               # no bias
        pw_w = pw_layer.get_weights()
        dw_shape = dw_w[0].shape
        pw_shape = pw_w[0].shape
        bias_shape = pw_w[1].shape

    mu, sigma = 0, 0.1
    grid = model.output_shape[1:3]
    grid_area = grid[0] * grid[1]

    dw_kernel = np.random.normal(mu, sigma, size=dw_shape) / grid_area
    pw_kernel = np.random.normal(mu, sigma, size=pw_shape) / grid_area
    bias = np.random.normal(mu, sigma, size=bias_shape) / grid_area

    if len(layers) == 1:
        layers[0].set_weights([dw_kernel, pw_kernel, bias])
    else:
        layers[0].set_weights([dw_kernel])
        layers[1].set_weights([pw_kernel, bias])

    return model


# ============================================================
# SELF TEST
# ============================================================

def _compare_against_stock():
    """
    Build both models and report the differences that matter.

    The two things to confirm: the output grid is 14x14, and the
    parameter count is unchanged. A different parameter count would mean
    something other than the stride changed by accident.
    """
    from akida_models import yolo_base

    print()
    print("Stock yolo_base vs stride-16 variant, 224x224x3, alpha 0.5")
    print("=" * 66)

    stock = yolo_base(input_shape=(224, 224, 3), classes=1,
                      nb_box=5, alpha=0.5)
    mine = yolo_base_stride16(input_shape=(224, 224, 3), classes=1,
                              nb_box=5, alpha=0.5)

    print(f"  stock  output {str(stock.output_shape):22s} "
          f"params {stock.count_params():,}")
    print(f"  s16    output {str(mine.output_shape):22s} "
          f"params {mine.count_params():,}")

    ok_grid = mine.output_shape[1:3] == (14, 14)
    ok_params = mine.count_params() == stock.count_params()

    print()
    print(f"  14x14 grid          : {'yes' if ok_grid else 'NO'}")
    print(f"  params unchanged    : {'yes' if ok_params else 'NO'}")

    # Where the activations actually grew.
    print()
    print("  Activation shapes, last four layers:")
    for m, label in ((stock, "stock"), (mine, "s16  ")):
        shapes = [l.output_shape for l in m.layers[-8:]
                  if hasattr(l, "output_shape")]
        tail = [s for s in shapes if isinstance(s, tuple) and len(s) == 4]
        print(f"    {label}: {[s[1:] for s in tail[-4:]]}")

    print()
    if ok_grid and ok_params:
        print("  Structurally correct. Next: run it through quantize ->")
        print("  convert -> map to see whether Akida accepts the larger")
        print("  activations in separable_12 and separable_13.")
    else:
        print("  Something other than the stride changed. Do not train")
        print("  on this until the parameter count matches.")


if __name__ == "__main__":
    _compare_against_stock()