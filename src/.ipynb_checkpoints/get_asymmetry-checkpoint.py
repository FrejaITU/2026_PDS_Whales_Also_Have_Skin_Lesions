import numpy as np
from skimage.transform import rotate


def get_asymmetry(image, mask):
    mask = np.asarray(mask) > 0

    scores = []

    for _ in range(6):
        segment = crop(mask)

        if segment is not None:
            flipped = np.fliplr(segment)

            difference = np.logical_xor(segment, flipped)
            union = np.logical_or(segment, flipped)

            score = np.sum(difference) / np.sum(union)
            scores.append(score)

        mask = rotate(mask, 30, order=0) > 0.5

    if len(scores) == 0:
        return {
            "avg_asymmetry_score": np.nan,
            "worst_score": np.nan
        }

    return {
        "avg_asymmetry_score": sum(scores) / len(scores),
        "worst_score": max(scores)
    }


def crop(mask):
    y, x = np.nonzero(mask)

    if len(y) == 0:
        return None

    return mask[y.min():y.max() + 1, x.min():x.max() + 1]