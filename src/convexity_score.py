import numpy as np
from skimage.morphology import convex_hull_image


def convexity_score(image, mask):
    mask = np.asarray(mask) > 0

    hull = convex_hull_image(mask)

    lesion_area = np.count_nonzero(mask)
    convex_hull_area = np.count_nonzero(hull)

    if convex_hull_area == 0:
        return {
            "convexity_score": np.nan,
        }

    convexity = lesion_area / convex_hull_area

    return {
        "convexity_score": convexity,
    }