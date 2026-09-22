"""
Start the backbone from ImageNet instead of from noise.

Why
---
The detector currently learns everything from 90 drone clips: edge
detectors, texture filters, and the notion of a drone, all at once. With
that little data it is cheap for the network to latch onto something
specific to a clip -- a particular sky, a particular horizon -- and the
training curve shows exactly that, with validation recall peaking at
epoch 2 and falling after.

AkidaNet has ImageNet weights available at the same alpha. Those layers
already hold edge and texture filters learned from 1.2M photographs.
Loading them leaves the network less to invent, and less room to
memorise.

What transfers, and what does not
---------------------------------
The backbone transfers. The ImageNet classifier head does not -- it
predicts 1000 classes and is discarded. The YOLO head that replaces it
starts random, as it must.

The stride change does not matter. A stride is a property of how a
kernel is applied, not of the kernel, so separable_12's weights have the
same shape at stride 1 as at stride 2 and load unchanged.

A caveat worth keeping in mind
------------------------------
A drone here is about 11 px and fits inside one feature cell after the
backbone. The features that can still see it come from the EARLY layers,
where spatial detail survives. ImageNet's biggest advantage is in the
deep, semantic layers -- which is where the drone has already been
averaged away. So expect a real gain, but a smaller one than transfer
learning usually gives.

Usage
-----
    from pretrained import load_imagenet_backbone
    n = load_imagenet_backbone(model, alpha=0.5)
"""

import numpy as np


def load_imagenet_backbone(model, alpha=0.5, verbose=True):
    """
    Copy ImageNet weights into every layer of `model` that has a
    same-named, same-shaped counterpart in pretrained AkidaNet.

    Returns the number of layers loaded. Layers with no counterpart --
    the YOLO head above all -- keep their initial values.

    Nothing is copied unless the shapes match exactly. A silent partial
    load would leave the model looking trained while carrying noise in
    half its filters, which is the worst way for this to fail.
    """
    from akida_models import akidanet_imagenet_pretrained

    # quantized=False: these are for float training, not for deployment.
    # The quantized copies are 4-bit and would be a poor starting point
    # for gradient descent.
    source = akidanet_imagenet_pretrained(alpha=alpha, quantized=False)

    by_name = {l.name: l for l in source.layers if l.weights}

    loaded = []
    skipped = []
    clashed = []

    for layer in model.layers:
        if not layer.weights:
            continue

        src = by_name.get(layer.name)

        if src is None:
            skipped.append(layer.name)
            continue

        want = [tuple(w.shape) for w in layer.weights]
        have = [tuple(w.shape) for w in src.weights]

        if want != have:
            clashed.append((layer.name, want, have))
            continue

        layer.set_weights(src.get_weights())
        loaded.append(layer.name)

    if verbose:
        print()
        print(f"IMAGENET BACKBONE  (alpha {alpha})")
        print(f"  loaded    : {len(loaded)} layers")
        print(f"  untouched : {len(skipped)} layers "
              "(the YOLO head, which has to start fresh)")

        for n in skipped:
            print(f"              {n}")

        if clashed:
            print(f"  SHAPE CLASH: {len(clashed)}")
            for n, want, have in clashed[:5]:
                print(f"              {n}: model {want} vs imagenet {have}")
            print()
            print("  A clash means alpha does not match the pretrained")
            print("  model, so those layers kept their random values.")

        print()

    return len(loaded)