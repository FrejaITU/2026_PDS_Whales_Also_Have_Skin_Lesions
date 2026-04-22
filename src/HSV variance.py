#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import nan
from statistics import variance
from skimage.color import rgb2hsv
from skimage.segmentation import slic
from scipy.stats import circmean, circvar

data_path = '../data/'
img_path = data_path + "imgs/"
mask_path = data_path + "masks/"
def load_image_and_mask(image_id, data_path=data_path):
    file_im = img_path + image_id
    file_mask = (mask_path + image_id).replace(".png", "_mask.png")
    im = plt.imread(file_im)
    mask = plt.imread(file_mask)
    
    return im, mask


metadata = pd.read_csv(data_path+"metadata.csv")
def slic_segmentation(
    image,
    mask,
    compactness=0.1,
    sigma=1,
    enforce_connectivity=True
):
    n_segments = max(15, min(100, round(mask.sum() / 400))) # There is a superpixel per at less 400 pixel.

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        mask=mask,
        start_label=1,
        channel_axis=-1,
        convert2lab=True,
        enforce_connectivity=enforce_connectivity
    )

    return segments

def get_hsv_means(image, slic_segments):
    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        # Create masked image where only specific segment is active
        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        #Get average HSV values from segment
        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') # Compute circular hue mean
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i)) # Compute saturation mean
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) # Compute value mean

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)

    return hsv_means

def hsv_var(image, slic_segments):
    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    # Compute variance for each channel seperately
    hue_var = circvar(hue, high=1, low=0)
    sat_var = variance(sat, sum(sat)/n)
    val_var = variance(val, sum(val)/n)

    return hue_var, sat_var, val_var

def hsv_var_score(image, mask):
    segments = slic_segmentation(image, mask)
    hvar, svar, vvar = hsv_var(image, segments)

    return({
        "hue_variance": hvar,
        "saturation_variance": svar,
        "value_variance": vvar
    })

