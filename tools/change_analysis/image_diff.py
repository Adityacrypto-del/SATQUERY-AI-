import numpy as np


def compute_difference(image_t1, image_t2):
    """
    Compute absolute pixel-wise difference between two
    spatially aligned images.
    """

    if image_t1.shape != image_t2.shape:
        raise ValueError(
            "T1 and T2 images must have the same shape."
        )

    return np.abs(
        image_t2.astype(np.float32)
        - image_t1.astype(np.float32)
    )


def create_change_mask(image_t1, image_t2, threshold=30):
    """
    Create a simple binary change mask from pixel differences.

    This is a deterministic supporting tool and does not replace
    DeltaVLM's semantic temporal reasoning.
    """

    difference = compute_difference(image_t1, image_t2)

    if difference.ndim == 3:
        difference = difference.mean(axis=2)

    return difference > threshold
