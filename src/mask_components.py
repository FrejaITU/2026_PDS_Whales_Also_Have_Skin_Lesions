### Mask components
import numpy as np
from skimage.measure import label


def mask_components_score(image, mask):
### we could use label for later use, if we decide to split images into components
    _, num = label(mask, connectivity=2, return_num=True)
    return {"mask_components": num}