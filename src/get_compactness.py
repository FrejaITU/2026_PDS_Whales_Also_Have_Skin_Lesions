import numpy as np
from skimage import morphology

def get_compactness(image, mask):
    """Gives polsby-popper score, how much does the shape deviate from a perfect circle."""
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask ^ mask_eroded)

    compact = perimeter**2 / (4 * np.pi * area)
    return {
        "Polsby-Popper": compact
    }