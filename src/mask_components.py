### Mask components
import numpy as np
from skimage.measure import label


def mask_components_score(image, mask):
    if not np.any(mask):
        return np.nan

    _, num = label(mask, connectivity=2, return_num=True)
    return num