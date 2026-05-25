#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from skimage.measure import regionprops
from skimage.transform import rotate
import matplotlib.pyplot as plt

mask = plt.imread("../data/masks_biggest_component/PAT_1855_3641_327_mask.png") > 0

def get_asymmetry(image, mask):
    mask = np.asarray(mask) > 0
    mask = resize_mask(mask)
    mask = center_mask(mask)

    scores = []

    for _ in range(6):
        plt.imshow(mask)
        plt.show()

        if segment is not None:
            flipped = np.fliplr(mask)

            difference = np.logical_xor(mask, flipped)
            union = np.logical_or(mask, flipped)

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


def resize_mask(mask):
    mask = crop(mask)
    h, w = mask.shape
    hw = max((h, w))*2
    extra = np.zeros((hw, hw))
    extra[0:h, 0:w] = mask
    
    return(extra)

def center_mask(mask):
    props = regionprops(mask.astype(int))[0]
    Yc, Xc = props.centroid


    cy, cx = np.array(mask.shape) / 2

    shift_y = cy - Yc
    shift_x = cx - Xc

    centered = shift(mask, shift=(shift_y, shift_x), order=0)

    return(centered > 0)


# In[ ]:




